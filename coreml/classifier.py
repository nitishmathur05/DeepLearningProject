import turicreate as turi
import os

def getImageFromPath(path):
    # norm path will noramilize the path /a/b/c/cat/meow1.png
    # dirname will return directoriles only /a/b/c/cat
    # basename cat
    return os.path.basename(os.path.dirname(os.path.normpath(path)))

myPath = '/Users/nitishmathur/Unimelb/Computing project/Original_Frame'
data = turi.image_analysis.load_images(myPath, with_path = True, recursive = True)

data["frames"] = data["path"].apply(lambda path: getImageFromPath(path))

print(data.groupby("frames",[turi.aggregate.COUNT]))

data.save("frames.sframe")

#data.explore()

train_data, test_data = data.random_split(0.9)

model = turi.image_classifier.create(train_data, target="frames")

predicitions = model.predict(test_data)

metrics = model.evaluate(test_data)

print(metrics["accuracy"])

print("Saving model")
model.save("frames.model")
print("Saving core ml model")
model.export_coreml("frames.mlmodel")
print("Done")