import pandas as pd
import pyodbc
from surprise import Dataset, Reader, SVD
from surprise.model_selection import train_test_split
from surprise import accuracy
import joblib

# ----------------------------------------------
# 1. ดึงข้อมูลจาก SQL Server
# ----------------------------------------------
# def fetch_data_from_sql():
#     # กำหนดค่าการเชื่อมต่อ SQL Server
#     conn = pyodbc.connect(
#         'DRIVER={ODBC Driver 17 for SQL Server};'
#         'SERVER=your_server_name;'  # แทนที่ด้วยชื่อหรือ IP ของเซิร์ฟเวอร์
#         'DATABASE=your_database_name;'  # ชื่อฐานข้อมูล
#         'UID=your_username;'  # ชื่อผู้ใช้งาน
#         'PWD=your_password;'  # รหัสผ่าน
#     )
#     # Query ข้อมูล
#     query = """
#     SELECT EmployeeID, CourseName, TrainingScore, TrainingDate
#     FROM TrainingRecords
#     """
#     data = pd.read_sql_query(query, conn)
#     conn.close()
#     return data

# # ดึงข้อมูลจาก SQL Server
# df = fetch_data_from_sql()

# ข้อมูลตัวอย่าง
data = {
    'EmployeeID': [1, 1, 2, 2, 3],
    'CourseName': ['Data Analysis', 'Machine Learning', 'Data Analysis', 'Deep Learning', 'Machine Learning'],
    'TrainingScore': [90, 85, 80, 75, 88]
}


df = pd.DataFrame(data)

# ----------------------------------------------
# 2. เตรียมข้อมูลสำหรับ Surprise
# ----------------------------------------------
# กำหนด Reader ให้เหมาะกับช่วงคะแนนของ TrainingScore
reader = Reader(rating_scale=(0, 100))
data_surprise = Dataset.load_from_df(df[['EmployeeID', 'CourseName', 'TrainingScore']], reader)

# ----------------------------------------------
# 3. แบ่งข้อมูล Train/Test
# ----------------------------------------------
trainset, testset = train_test_split(data_surprise, test_size=0.25, random_state=42)

# ----------------------------------------------
# 4. สร้างและฝึกโมเดล SVD
# ----------------------------------------------
model = SVD()
model.fit(trainset)

# ----------------------------------------------
# 5. ทดสอบโมเดล
# ----------------------------------------------
predictions = model.test(testset)
print("RMSE:", accuracy.rmse(predictions))

# ----------------------------------------------
# 6. บันทึกโมเดล
# ----------------------------------------------
joblib.dump(model, 'employee_recommend_model.pkl')
print("Model saved as 'employee_recommend_model.pkl'")

# ----------------------------------------------
# 7. ฟังก์ชันแนะนำคอร์ส
# ----------------------------------------------
def recommend_courses(employee_id, all_courses, model, df):
    # หาคอร์สที่พนักงานยังไม่ได้เรียน
    learned_courses = df[df['EmployeeID'] == employee_id]['CourseName'].unique()
    not_learned_courses = [course for course in all_courses if course not in learned_courses]
    
    # ทำนายคะแนนสำหรับคอร์สที่ยังไม่ได้เรียน
    recommendations = []
    for course in not_learned_courses:
        pred = model.predict(employee_id, course)
        recommendations.append((course, pred.est))
    
    # เรียงลำดับตามคะแนนที่ทำนาย
    recommendations.sort(key=lambda x: x[1], reverse=True)
    return recommendations

# ----------------------------------------------
# 8. แสดงคำแนะนำสำหรับพนักงาน
# ----------------------------------------------
# รายชื่อคอร์สทั้งหมด
all_courses = df['CourseName'].unique()

# แนะนำคอร์สสำหรับพนักงานคนที่ 1
employee_id = 1  # รหัสพนักงานที่ต้องการคำแนะนำ
recommendations = recommend_courses(employee_id, all_courses, model, df)
print(f"Recommendations for Employee {employee_id}:", recommendations)
