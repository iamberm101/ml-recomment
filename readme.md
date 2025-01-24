git init
git add .
git commit -m "Initial commit: Resume screening ML service"
git branch -M main
git remote add origin https://github.com/iamberm101/ml-recomment.git
git push -u origin main


# Resume Screening API
- FastAPI service for resume screening using ML
- Integrates with SQL Server
- Includes API key authentication
- Auto-retrains weekly

## Setup
1. Install requirements
2. Configure database connection
3. Run: uvicorn api:app --reload

requirements.txt เป็นไฟล์ที่ระบุ package ที่จำเป็นสำหรับโปรเจค ช่วยให้:

ติดตั้ง dependencies ได้ง่ายด้วยคำสั่ง: pip install -r requirements.txt
ผู้อื่นรู้ว่าต้องติดตั้ง package อะไรบ้าง
ระบุเวอร์ชันที่แน่นอนของแต่ละ package เพื่อป้องกันปัญหาความเข้ากันไม่ได้