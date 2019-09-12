const functions = require('firebase-functions');
const gcs = require('@google-cloud/storage')();
const os = require('os');
const path = require('path');
// // Create and Deploy Your First Cloud Functions
// // https://firebase.google.com/docs/functions/write-firebase-functions
//
// exports.helloWorld = functions.https.onRequest((request, response) => {
//  response.send("Hello from Firebase!");
// });

exports.myfirstcouldfunc = functions.storage.object().onFinalize(event => {
	const object = event.data;
	const bucket = object.bucket;
	const contentType = object.contentType;
	const filePath = object.name;
	
	console.log("Hello from nitish");

	if (path.basename(filePath).startsWith('sensored-image')) {
		console.log("File already processed");
		return;
	}

	const destBucket = gcs.bucket(bucket);
	const tmpFilePath = path.join(os.tmpdir(), path.basename(filePath));
	const metadata = { contentType: contentType };

	return destBucket.file(filePath).download({
		destination : tmpFilePath
	}).then( () => {
		return destBucket.upload(tmpFilePath, {
			destination : 'sensored-image' + path.basename(filePath),
			metadata : metadata
		})
	});
});

