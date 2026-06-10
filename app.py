from datetime import datetime
from functools import wraps
import json
import os
import sys
import logging
import threading
import time
import uuid
import webbrowser

import cv2
from flask import Flask, Response, jsonify, redirect, render_template, request, session, url_for, flash
from ultralytics import YOLO
from werkzeug.security import check_password_hash, generate_password_hash
from werkzeug.utils import secure_filename


logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(os.path.join(os.path.abspath("."), "app.log"), encoding="utf-8"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)

# Lock bảo vệ tránh race condition khi nhiều user ghi file history cùng lúc
_history_lock = threading.Lock()


def resource_path(relative_path):
    """
    Lay duong dan dung khi chay bang Python hoac khi dong goi thanh .exe bang PyInstaller.
    """
    try:
        base_path = sys._MEIPASS
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)


app = Flask(
    __name__,
    template_folder=resource_path("templates"),
    static_folder=resource_path("static"),
)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "do-an-ai-demo-secret-key")

UPLOAD_FOLDER = resource_path("static/uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # Giới hạn file upload tối đa 16 MB

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif", ".avif", ".tiff", ".tif"}

USERS_FILE = resource_path("users.json")
HISTORY_FILE = resource_path("classification_history.json")

# Giu nguyen duong dan model YOLOv8/best.pt hien co trong source.
model_path = resource_path(r"models\best.pt")
model = YOLO(model_path)


def load_json_file(path, default_value):
    if not os.path.exists(path):
        return default_value

    try:
        with open(path, "r", encoding="utf-8") as file:
            return json.load(file)
    except (json.JSONDecodeError, OSError):
        return default_value


def save_json_file(path, data):
    with open(path, "w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def login_required(view_func):
    @wraps(view_func)
    def wrapped_view(*args, **kwargs):
        if "username" not in session:
            flash("Vui lòng đăng nhập để sử dụng hệ thống.", "warning")
            return redirect(url_for("login"))
        return view_func(*args, **kwargs)

    return wrapped_view


def get_users():
    return load_json_file(USERS_FILE, {})


def save_users(users):
    save_json_file(USERS_FILE, users)


def get_history():
    return load_json_file(HISTORY_FILE, [])


def save_history(history):
    save_json_file(HISTORY_FILE, history)


def append_to_history(item):
    """Thêm một bản ghi vào lịch sử một cách an toàn (thread-safe)."""
    with _history_lock:
        history = load_json_file(HISTORY_FILE, [])
        history.append(item)
        save_json_file(HISTORY_FILE, history)


def class_metadata(raw_class_name):
    raw = (raw_class_name or "").lower()

    if "khong" in raw and "tai" in raw and "che" in raw:
        return {
            "label": "Không thể tái chế",
            "status": "not_recyclable",
            "draw_label": "Khong tai che",
            "box_color": (0, 0, 255),
            "suggestion": "Bỏ vào thùng rác thông thường, tránh trộn lẫn với nhóm rác tái chế.",
        }

    if "tai" in raw and "che" in raw:
        return {
            "label": "Có thể tái chế",
            "status": "recyclable",
            "draw_label": "Tai che",
            "box_color": (0, 180, 0),
            "suggestion": "Làm rỗng chai, bóp gọn nếu cần và bỏ vào thùng rác tái chế.",
        }

    return {
        "label": raw_class_name or "Không xác định",
        "status": "unknown",
        "draw_label": raw_class_name or "Unknown",
        "box_color": (120, 120, 120),
        "suggestion": "Kiểm tra lại vật thể hoặc chụp rõ hơn để hệ thống nhận diện chính xác.",
    }


def no_detection_metadata():
    return {
        "label": "Không phát hiện",
        "status": "unknown",
        "suggestion": "Thử chụp rõ vật thể hơn, đặt chai/rác ở giữa khung hình và nền ít nhiễu.",
    }


@app.route("/")
def home():
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        users = get_users()
        user = users.get(username)

        if user and check_password_hash(user["password"], password):
            session["username"] = username
            session["full_name"] = user.get("full_name") or username
            logger.info("Đăng nhập thành công: user=%s", username)
            flash("Đăng nhập thành công.", "success")
            return redirect(url_for("dashboard"))

        flash("Tên đăng nhập hoặc mật khẩu không đúng.", "error")

    return render_template("login.html")


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        full_name = request.form.get("full_name", "").strip()
        username = request.form.get("username", "").strip()
        password = request.form.get("password", "")
        confirm_password = request.form.get("confirm_password", "")
        users = get_users()

        if len(username) < 3:
            flash("Tên đăng nhập cần có ít nhất 3 ký tự.", "error")
        elif len(password) < 4:
            flash("Mật khẩu cần có ít nhất 4 ký tự.", "error")
        elif password != confirm_password:
            flash("Mật khẩu xác nhận chưa khớp.", "error")
        elif username in users:
            flash("Tên đăng nhập đã tồn tại.", "error")
        else:
            users[username] = {
                "full_name": full_name or username,
                "password": generate_password_hash(password),
                "created_at": datetime.now().strftime("%d/%m/%Y %H:%M"),
            }
            save_users(users)
            flash("Tạo tài khoản thành công. Bạn có thể đăng nhập ngay.", "success")
            return redirect(url_for("login"))

    return render_template("register.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Bạn đã đăng xuất.", "success")
    return redirect(url_for("login"))


@app.route("/dashboard")
@login_required
def dashboard():
    return render_template("index.html", active_page="classify")


@app.route("/history")
@login_required
def history_page():
    username = session.get("username")
    history_items = [item for item in get_history() if item.get("username") == username]
    history_items.reverse()

    stats = {
        "total": len(history_items),
        "recyclable": sum(1 for item in history_items if item.get("status") == "recyclable"),
        "not_recyclable": sum(1 for item in history_items if item.get("status") == "not_recyclable"),
    }

    return render_template(
        "history.html",
        active_page="history",
        history_items=history_items,
        stats=stats,
    )


@app.route("/classify", methods=["POST"])
def classify():
    if "username" not in session:
        return jsonify({"error": "Vui lòng đăng nhập để phân loại ảnh."}), 401

    if "file" not in request.files:
        return jsonify({"error": "Không có file nào được tải lên"}), 400

    file = request.files["file"]

    if file.filename == "":
        return jsonify({"error": "Chưa chọn file"}), 400

    safe_name = secure_filename(file.filename) or "upload.jpg"
    ext = os.path.splitext(safe_name)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        return jsonify({"error": "Định dạng file không được hỗ trợ. Vui lòng upload ảnh JPG, PNG, WEBP, ..."}), 400

    filename = f"{int(time.time())}_{uuid.uuid4().hex[:8]}_{safe_name}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    try:
        start_time = time.time()
        results = model.predict(filepath, imgsz=640, conf=0.25, verbose=False)
        result = results[0]
        total_time = time.time() - start_time

        image = cv2.imread(filepath)

        if image is None:
            return jsonify({"error": "Không đọc được ảnh đã tải lên"}), 400

        predictions = []

        for box, cls, conf in zip(result.boxes.xyxy, result.boxes.cls, result.boxes.conf):
            x1, y1, x2, y2 = map(int, box)
            raw_class = result.names[int(cls)]
            confidence = float(conf)
            metadata = class_metadata(raw_class)

            cv2.rectangle(image, (x1, y1), (x2, y2), metadata["box_color"], 2)
            cv2.putText(
                image,
                f"{raw_class} {confidence * 100:.1f}%",
                (x1, max(y1 - 10, 20)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                metadata["box_color"],
                2,
            )

            predictions.append(
                {
                    "raw_class": raw_class,
                    "confidence": confidence,
                    **metadata,
                }
            )

        out_filename = f"out_{filename}"
        out_path = os.path.join(app.config["UPLOAD_FOLDER"], out_filename)
        cv2.imwrite(out_path, image)

        if predictions:
            top = max(predictions, key=lambda item: item["confidence"])
            pred_class = top["label"]
            raw_class = top["raw_class"]
            confidence_percent = round(top["confidence"] * 100, 2)
            confidence = f"{confidence_percent:.2f}%"
            status = top["status"]
            suggestion = top["suggestion"]
        else:
            empty = no_detection_metadata()
            pred_class = empty["label"]
            raw_class = "no_detection"
            confidence_percent = 0
            confidence = "0.00%"
            status = empty["status"]
            suggestion = empty["suggestion"]

    except Exception as error:
        logger.error("Lỗi khi phân loại ảnh: %s", error, exc_info=True)
        return jsonify({"error": str(error)}), 500
    finally:
        # Xóa file ảnh gốc sau khi xử lý, chỉ giữ ảnh kết quả annotated
        try:
            if os.path.exists(filepath):
                os.remove(filepath)
        except OSError:
            pass

    result_image_url = url_for("static", filename=f"uploads/{out_filename}")
    created_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    history_item = {
        "id": uuid.uuid4().hex,
        "username": session.get("username"),
        "image": result_image_url,
        "class": pred_class,
        "raw_class": raw_class,
        "confidence": confidence,
        "confidence_percent": confidence_percent,
        "status": status,
        "suggestion": suggestion,
        "inference_time": f"{total_time:.3f} giây",
        "created_at": created_at,
    }

    append_to_history(history_item)
    logger.info(
        "Phân loại thành công: user=%s, class=%s, conf=%s",
        session.get("username"), raw_class, confidence,
    )

    return jsonify(
        {
            "class": pred_class,
            "raw_class": raw_class,
            "confidence": confidence,
            "confidence_percent": confidence_percent,
            "image": result_image_url,
            "time": f"{total_time:.3f} giây",
            "suggestion": suggestion,
            "status": status,
            "created_at": created_at,
        }
    )


def gen_frames():
    # Thử lần lượt các camera index 0-4 để tìm webcam khả dụng
    cap = None
    for cam_index in range(5):
        _cap = cv2.VideoCapture(cam_index, cv2.CAP_DSHOW)
        if _cap.isOpened():
            ret, _ = _cap.read()
            if ret:
                cap = _cap
                logger.info("Mở webcam thành công ở index %d.", cam_index)
                break
        _cap.release()

    if cap is None:
        logger.error("Không tìm thấy webcam khả dụng (đã thử index 0-4).")
        return

    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)

    frame_count = 0
    last_annotated = None

    try:
        while True:
            success, frame = cap.read()

            if not success or frame is None:
                logger.warning("Không đọc được frame từ webcam.")
                break

            frame_count += 1

            # Chi nhan dien moi 10 frame de giam lag khi demo webcam realtime.
            if frame_count % 10 == 0:
                try:
                    results = model.predict(
                        frame,
                        imgsz=416,
                        conf=0.25,
                        verbose=False,
                    )
                    last_annotated = results[0].plot()
                except Exception as error:
                    logger.error("Lỗi khi nhận diện camera: %s", error)
                    last_annotated = frame

            display_frame = last_annotated if last_annotated is not None else frame
            ret, buffer = cv2.imencode(".jpg", display_frame)

            if not ret:
                continue

            frame_bytes = buffer.tobytes()

            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
            )

    finally:
        cap.release()
        logger.info("Đã đóng webcam.")


@app.route("/video_feed")
@login_required
def video_feed():
    return Response(
        gen_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


@app.errorhandler(404)
def page_not_found(error):
    flash("Trang không tồn tại.", "error")
    if "username" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({"error": "File quá lớn. Vui lòng upload ảnh dưới 16 MB."}), 413


def open_browser():
    webbrowser.open_new("http://127.0.0.1:5000")


if __name__ == "__main__":
    threading.Timer(2.0, open_browser).start()
    app.run(
        host="127.0.0.1",
        port=5000,
        debug=False,
        use_reloader=False,
    )
