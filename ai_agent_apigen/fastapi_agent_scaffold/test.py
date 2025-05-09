from pymongo import MongoClient

client = MongoClient("mongodb://mongoAdmin:securePass123@3.105.47.11:27017/?authSource=admin")
db = client["sna_database"]
collection = db["ClassB"]

collection.insert_many([
    {"attendance": "Present", "date": "2024-05-01"},
    {"attendance": "Absent", "date": "2024-05-02"},
    {"attendance": "Present", "date": "2024-05-03"}
])

print("Sample data inserted.")