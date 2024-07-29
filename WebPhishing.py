# Chương trình phát hiện Web Phishing thông qua phân tích Bố cục trang web dùng hai thuật toán học máy SVM và DecisionTree.
import csv
import datetime
import logging
import os
import tkinter as tk
import warnings
from concurrent.futures import ThreadPoolExecutor
import pandas as pd
import requests
from bs4 import BeautifulSoup
from joblib import dump, load
from sklearn.metrics import accuracy_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import VotingClassifier

# Các hằng số
TRAINING_DIR = 'training'
VALIDATION_DIR = 'validation'
PHISHING_DIR = 'Phishing'
NOT_PHISHING_DIR = 'NotPhishing'
SVM_MODEL_PATH = 'svm_model.pkl'
DT_MODEL_PATH = 'dt_model.pkl'
LOG_FILE = 'wp.log'

# Thiết lập logging
logging.basicConfig(
    format='%(asctime)s - %(levelname)s - %(message)s',
    level=logging.INFO,
    filename=LOG_FILE
)

console = logging.StreamHandler()
console.setLevel(logging.INFO)
logging.getLogger('').addHandler(console)
warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")


# Hàm đọc HTML từ file
def read_html_from_file(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            content = f.read()
            logging.info(f"Đọc xong file HTML {filepath}")
            return content
    except Exception as e:
        logging.warning(f"Không thể đọc file HTML {filepath}. Lỗi: {e}")
        return None


# Hàm đọc dữ liệu từ các thư mục training và validation
def read_data(directory):
    categories = ['Phishing', 'NotPhishing']
    data = []
    for category in categories:
        category_dir = os.path.join(directory, category)
        for filename in os.listdir(category_dir):
            filepath = os.path.join(category_dir, filename)
            data.append({'path': filepath, 'category': category})
    return pd.DataFrame(data)


# Đọc dữ liệu từ thư mục training và validation
training_data = read_data(TRAINING_DIR)
validation_data = read_data(VALIDATION_DIR)
logging.info('Đã đọc xong dữ liệu training và validation')

# Đọc HTML từ các file
with ThreadPoolExecutor(max_workers=10) as executor:
    training_data['html'] = list(executor.map(read_html_from_file, training_data['path']))
    validation_data['html'] = list(executor.map(read_html_from_file, validation_data['path']))


# Hàm rút trích các đặc trưng về bố cục trang web từ tệp HTML
def extract_features_from_html(html):
    try:
        soup = BeautifulSoup(html, 'html.parser')
    except Exception as e:
        logging.warning(f'Lỗi khi phân tích tệp HTML: {e}')
        return None

    tags_to_count = ['a', 'b', 'body', 'div', 'embed', 'form', 'head', 'html', 'i', 'iframe', 'img', 'input',
                     'link', 'meta', 'p', 'script', 'strong', 'style', 'title', 'u']
    element_counts = {tag: 0 for tag in tags_to_count}
    for tag in soup.find_all(tags_to_count):
        element_counts[tag.name] += 1

    internal_links = sum(1 for a in soup.find_all('a') if 'href' in a.attrs and a['href'].startswith('/'))
    external_links = sum(1 for a in soup.find_all('a') if 'href' in a.attrs and a['href'].startswith('http'))
    insecure_links = sum(1 for a in soup.find_all('a') if 'href' in a.attrs and a['href'].startswith('http://'))
    external_images = sum(1 for img in soup.find_all('img') if 'src' in img.attrs and img['src'].startswith('http'))
    insecure_forms = sum(
        1 for form in soup.find_all('form') if 'action' in form.attrs and not form['action'].startswith('/'))

    dom_complexity = len(soup.contents)
    duplicated_content = len([tag for tag in soup.stripped_strings if list(soup.stripped_strings).count(tag) > 1])

    return list(element_counts.values()) + [internal_links, external_links, insecure_links, external_images,
                                            insecure_forms, dom_complexity, duplicated_content]


# Hàm xử lý dữ liệu
def process_data(df):
    df['html'] = df['path'].apply(read_html_from_file)
    df['features'] = df['html'].apply(extract_features_from_html)
    df = df.dropna(subset=['features'])
    features = pd.DataFrame(df['features'].to_list())
    features.columns = features.columns.astype(str)
    df = pd.concat([df, features], axis=1)
    df = df.drop(columns=['html', 'features', 'path'])
    return df


# Xử lý dữ liệu training và validation
training_data = process_data(training_data)
training_data = training_data.dropna()
validation_data = process_data(validation_data)
validation_data = validation_data.dropna()
logging.info('Đã xử lý xong dữ liệu training và validation')


# Hàm để lưu mô hình ML vào tệp
def save_model(model, filename):
    try:
        dump(model, filename)
    except Exception as e:
        logging.warning(f"Không thể lưu mô hình. Lỗi: {e}")


# Hàm để tải mô hình ML từ tệp
def load_model(filename):
    if not os.path.exists(filename):
        logging.warning("File mô hình không tồn tại: " + filename)
        return None
    try:
        return load(filename)
    except Exception as e:
        logging.warning(f"Không thể tải mô hình từ file {filename}. Lỗi: {e}")


# Tạo DataFrame từ danh sách các file HTML phishing và không phishing trong thư mục training
training_phishing_data = training_data[training_data['category'] == 'Phishing'].copy()
training_not_phishing_data = training_data[training_data['category'] == 'NotPhishing'].copy()

# Gán nhãn cho dữ liệu training
training_phishing_data['label'] = 1
training_not_phishing_data['label'] = 0

# Kết hợp dữ liệu training
training_data = pd.concat([training_phishing_data, training_not_phishing_data])

# Tạo DataFrame từ danh sách các file HTML phishing và không phishing trong thư mục validation
validation_phishing_data = validation_data[validation_data['category'] == 'Phishing'].copy()
validation_not_phishing_data = validation_data[validation_data['category'] == 'NotPhishing'].copy()

# Gán nhãn cho dữ liệu validation
validation_phishing_data['label'] = 1
validation_not_phishing_data['label'] = 0

# Kết hợp dữ liệu validation
validation_data = pd.concat([validation_phishing_data, validation_not_phishing_data])

# Chia dữ liệu thành tập huấn luyện và tập kiểm tra
X_train = training_data.drop(columns=['label', 'category'])
y_train = training_data['label']
X_test = validation_data.drop(columns=['label', 'category'])
y_test = validation_data['label']

# Tải mô hình SVM
if os.path.exists(SVM_MODEL_PATH):
    clf_svm = load_model(SVM_MODEL_PATH)
    svm_accuracy = accuracy_score(y_test, clf_svm.predict(X_test)) * 100
    logging.info('Đã tải xong mô hình SVM')
else:
    svm = SVC(kernel='linear', probability=True)
    clf_svm = svm.fit(X_train, y_train)
    svm_pred = clf_svm.predict(X_test)
    svm_accuracy = accuracy_score(y_test, svm_pred) * 100

    # Lưu mô hình SVM
    save_model(clf_svm, SVM_MODEL_PATH)
    logging.info('Đã lưu xong mô hình SVM')

# Tải mô hình Decision Tree
if os.path.exists(DT_MODEL_PATH):
    clf_dt = load_model(DT_MODEL_PATH)
    dt_accuracy = accuracy_score(y_test, clf_dt.predict(X_test)) * 100
    logging.info('Đã tải xong mô hình Decision Tree')
else:
    dt = DecisionTreeClassifier()
    clf_dt = dt.fit(X_train, y_train)
    dt_pred = clf_dt.predict(X_test)
    dt_accuracy = accuracy_score(y_test, dt_pred) * 100

    # Lưu mô hình Decision Tree
    save_model(clf_dt, DT_MODEL_PATH)
    logging.info('Đã lưu xong mô hình Decision Tree')

# Hiển thị độ chính xác của hai mô hình
logging.info(f'Độ chính xác của mô hình SVM: {svm_accuracy}%')
logging.info(f'Độ chính xác của mô hình Decision Tree: {dt_accuracy}%')


# Hàm để tính ma trận nhầm lẫn
def calculate_confusion_matrix(model, X, y):
    y_pred = model.predict(X)
    return confusion_matrix(y, y_pred)


# Tính ma trận nhầm lẫn cho hai thuật toán
svm_confusion_matrix = calculate_confusion_matrix(clf_svm, X_test, y_test)
dt_confusion_matrix = calculate_confusion_matrix(clf_dt, X_test, y_test)

# Hiển thị ma trận nhầm lẫn cho hai thuật toán
logging.info("Ma trận nhầm lẫn cho SVM:")
logging.info(svm_confusion_matrix)
logging.info("\nMa trận nhầm lẫn cho Decision Tree:")
logging.info(dt_confusion_matrix)

# Tạo một voting classifier
voting_clf = VotingClassifier(
    estimators=[('svm', clf_svm), ('dt', clf_dt)],
    voting='soft'
)

# Huấn luyện voting classifier trên tập huấn luyện
voting_clf.fit(X_train, y_train)


# Hàm đọc HTML từ URL
def read_html_from_url(url):
    try:
        response = requests.get(url)
        if response.status_code == 200:
            content = response.text
            logging.info(f"Đọc xong tệp HTML từ URL: {url}")
            return content
        else:
            logging.warning(f"Lỗi khi lấy tệp HTML từ URL: {url}. Mã trạng thái: {response.status_code}")
            return None
    except Exception as e:
        logging.warning(f"Lỗi khi lấy tệp HTML từ URL: {url}. Lỗi: {e}")
        return None


# Hàm lấy tệp HTML từ URL và thực hiện phân tích
def process_url(url):
    html_content = read_html_from_url(url)
    if html_content is None:
        return "Không thể lấy tệp HTML từ URL nên không thể phân tích"

    features = extract_features_from_html(html_content)
    if features is None:
        return "Không thể rút trích đặc trưng từ tệp HTML"

    pred = voting_clf.predict([features])[0]

    if pred == 0:
        return "Tệp HTML từ URL cho thấy trang web KHÔNG CÓ dấu hiệu lừa đảo"
    elif pred == 1:
        return "Tệp HTML từ URL cho thấy trang web CÓ dấu hiệu lừa đảo"


# Hàm xử lý sự kiện khi muốn kiểm tra tệp HTML từ URL
def check_url():
    url = url_entry.get()
    if url:
        result = process_url(url)
        if result is None:
            logging.warning('Không thể lấy kết quả')
        else:
            if "KHÔNG CÓ" in result:
                result_label.configure(text=result, fg="green")
            else:
                result_label.configure(text=result, fg="red")

        with open('results.csv', 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow([url, result, datetime.datetime.now()])


# Lấy URL từ clipboard
def get_clipboard_url():
    try:
        import tkinter as tk
        root = tk.Tk()
        root.withdraw()
        url = root.clipboard_get()
        url_entry.delete(0, tk.END)
        url_entry.insert(0, url)
    except Exception as e:
        logging.warning(f"Lỗi khi lấy URL từ clipboard: {e}")


app = tk.Tk()
frame = tk.Frame(app)
clipboard_button = tk.Button(frame, text="Lấy URL", command=get_clipboard_url)
clipboard_button.configure(width=10, height=1)
clipboard_button.grid(row=3, column=1, padx=(0, 10), pady=(0, 10), sticky=tk.E)

submit_button = tk.Button(frame, text="Kiểm tra", command=check_url)
submit_button.configure(width=10, height=1)
submit_button.grid(row=4, column=0, columnspan=2)

result_label = tk.Label(frame, text="")
result_label.grid(row=5, column=0, columnspan=2, pady=(10, 0))

# Giao diện người dùng
app.title("Kiểm tra website thông qua phân tích Bố cục trang web")
app.geometry("600x250")

frame.place(relx=0.5, rely=0.5, anchor=tk.CENTER)

title_label = tk.Label(frame, text="ỨNG DỤNG KIỂM TRA WEBSITE", font=("Times New Roman", 16, "bold"))
title_label.grid(row=0, column=0, columnspan=2, pady=(0, 10))

title_label = tk.Label(frame, text="TÁC GIẢ: TRẦN THIỆN NHÂN", font=("Times New Roman", 14, "bold"))
title_label.grid(row=1, column=0, columnspan=2, pady=(0, 10))

url_label = tk.Label(frame, text="Nhập URL:")
url_label.grid(row=2, column=0, padx=(10, 5), pady=(10, 0), sticky=tk.W)

url_entry = tk.Entry(frame, width=50)
url_entry.grid(row=3, column=0, padx=(5, 10), pady=(0, 10), sticky=tk.N + tk.S)

app.mainloop()
