import os
import firebase_admin
from firebase_admin import credentials, firestore, storage





def hello_world():
	print ("Hello World")

















firebase_key = '../firebase_key.json'
bucket_name = 'gs://chatapplication-d7a3e.appspot.com/'

cred = credentials.Certificate(firebase_key)

default_app = firebase_admin.initialize_app(cred, {
    'storageBucket': bucket_name
})

db = firestore.client()

bucket = storage.bucket()
print ("Success!!")
print (bucket)