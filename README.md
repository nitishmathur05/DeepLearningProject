# DeepLearningProject

- Train
python network_trainer.py --image_dir /mnt/project/NPDI/Frames 

- Test
python test_network.py test-images\<img>

Steps to run the Application:

Step 1: Open the terminal and go to the ChatApplication Folder

Step 2: Install the required pods by running the following command in the terminal: pod install 
(If the command was not recognised, Cocoa-pods may not be installed on the computer. To get Cocoa-pods refer to: https://guides.cocoapods.org/using/getting-started.html)

Step 3: Open ChatApplication.xcworkspace within the ChatApplication Folder in Xcode 11.1 or later

Step 4: Select a Simulator or Device within Xcode.

Step 5: Build and Run the project by clicking the button with the play icon in the main menu.

Step 6: Done

Dependencies Used:

Alamofire: Alamofire is a Swift-based HTTP networking library for iOS and macOS. It is used in this application to consume the Nectar service endpoint. (https://github.com/Alamofire/Alamofire.git)

GeoFire: GeoFire is an open-source library that allows you to store and query a set of keys based on their geographic location. It is used in this application to perform geo-quires to view application users that are in close proximity of the local user. ('https://github.com/firebase/geofire-objc.git')

SwiftOverlays: It is a Swift-based library that is used to display blocking overlays. (https://github.com/peterprokop/SwiftOverlays.git)

lottie-ios: It is a library that renders After Effects animations in real time. It is used in this application to display animations. (https://github.com/airbnb/lottie-ios.git)
