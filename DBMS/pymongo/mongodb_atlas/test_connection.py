from pymongo import MongoClient

uri = "mongodb://sagardam:<password>@ac-snv03wl-shard-00-00.ryvurop.mongodb.net:27017,ac-snv03wl-shard-00-01.ryvurop.mongodb.net:27017,ac-snv03wl-shard-00-02.ryvurop.mongodb.net:27017/?ssl=true&replicaSet=atlas-xxxxxx-shard-0&authSource=admin&retryWrites=true&w=majority"

client = MongoClient(uri, serverSelectionTimeoutMS=10000)

try:
    client.admin.command('ping')
    print("Pinged your deployment. You successfully connected to MongoDB!")
except Exception as e:
    print("Connection failed:", e)
