from fastapi import FastAPI, HTTPException, Depends, Header
from pydantic import BaseModel
import joblib
import pandas as pd

# สร้างแอป FastAPI
app = FastAPI()

# โหลดโมเดลที่บันทึกไว้
model = joblib.load('employee_recommend_model.pkl')

# ตัวอย่างข้อมูลคอร์ส (จำลองจาก SQL หรือ Dataset)
data = {
    'EmployeeID': [1, 1, 2, 2, 3],
    'CourseName': ['Data Analysis', 'Machine Learning', 'Data Analysis', 'Deep Learning', 'Machine Learning'],
    'TrainingScore': [90, 85, 80, 75, 88]
}
df = pd.DataFrame(data)

# รายชื่อคอร์สทั้งหมด
all_courses = df['CourseName'].unique()

# ระบุ API Key ที่อนุญาต
API_KEY = "biosoft"

# ฟังก์ชันตรวจสอบ API Key
def verify_api_key(x_api_key: str = Header(...)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Unauthorized: Invalid API Key")

# โครงสร้างข้อมูลที่รับจาก API
class EmployeeData(BaseModel):
    EmployeeID: int

# Route ทดสอบ API
@app.get("/")
def read_root():
    return {"message": "Welcome to the Course Recommendation API"}

# Route สำหรับแนะนำคอร์ส
@app.post("/recommend", dependencies=[Depends(verify_api_key)])
def recommend(data: EmployeeData):
    employee_id = data.EmployeeID

    # หาคอร์สที่พนักงานยังไม่ได้เรียน
    learned_courses = df[df['EmployeeID'] == employee_id]['CourseName'].unique()
    not_learned_courses = [course for course in all_courses if course not in learned_courses]
    
    # ทำนายคะแนนสำหรับคอร์สที่ยังไม่ได้เรียน
    recommendations = []
    for course in not_learned_courses:
        pred = model.predict(employee_id, course)
        recommendations.append({"course": course, "predicted_score": pred.est})

    # เรียงลำดับตามคะแนนที่ทำนาย
    recommendations.sort(key=lambda x: x["predicted_score"], reverse=True)

    # หากไม่มีคอร์สแนะนำ
    if not recommendations:
        raise HTTPException(status_code=404, detail="No course recommendations available for this employee.")

    # ส่งผลลัพธ์กลับ
    return {
        "EmployeeID": employee_id,
        "recommendations": recommendations
    }
