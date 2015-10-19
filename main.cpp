#include <iostream>
#include <chrono>
#include <unistd.h>

#include <opencv2/opencv.hpp>

#include <../AAMlib/icaam.h>
#include <../AAMlib/robustaam.h>
#include <../AAMlib/trainingdata.h>

#define WINDOW_NAME "AAM-Example"

using namespace std;
using namespace cv;

//ICAAM aam;
RobustAAM aam;

//Parameters for the fitting
int numShapeParameters = 11;          //Number of used shape parameters
int numAppParameters = 20;          //Number of used appearance parameters
float fitThreshold = 0.05f;      //Termination condition

vector<string> descriptions;
Mat groups;

void loadTrainingData(string fileName) {
    TrainingData t;
    t.loadDataFromFile(fileName);

    Mat p = t.getPoints();
    Mat i = t.getImage();

    if(descriptions.empty()) {
        descriptions = t.getDescriptions();
        groups = t.getGroups();
    }

    i.convertTo(i, CV_32FC3);
    cvtColor(i,i,CV_BGR2GRAY);

    aam.addTrainingData(p, i);
}

Mat drawShape(Mat image, Mat points) {
    if(!aam.triangles.empty()) {
        for(int i=0; i<aam.triangles.rows; i++) {
            Point a,b,c;
            a = aam.getPointFromMat(points, aam.triangles.at<int>(i,0));
            b = aam.getPointFromMat(points, aam.triangles.at<int>(i,1));
            c = aam.getPointFromMat(points, aam.triangles.at<int>(i,2));

            line(image, a, b, Scalar(255,0,255),1);
            line(image, a, c, Scalar(255,0,255),1);
            line(image, b, c, Scalar(255,0,255),1);
        }
    }

    return image;
}

int main()
{
    string filePath_train = "/home/lucas/Dropbox/Diplomarbeit/Code/trainingdata/";
    string filePath_test = "/home/lucas/Dropbox/Diplomarbeit/Code/testimages/";

    loadTrainingData(filePath_train+"data1.xml");
    loadTrainingData(filePath_train+"data2.xml");
    loadTrainingData(filePath_train+"data3.xml");
    loadTrainingData(filePath_train+"data4.xml");
    loadTrainingData(filePath_train+"data5.xml");
    loadTrainingData(filePath_train+"data6.xml");
    loadTrainingData(filePath_train+"data7.xml");
    loadTrainingData(filePath_train+"data8.xml");
    loadTrainingData(filePath_train+"data9.xml");
    loadTrainingData(filePath_train+"data10.xml");
    loadTrainingData(filePath_train+"data11.xml");
    loadTrainingData(filePath_train+"data12.xml");
    loadTrainingData(filePath_train+"data13.xml");
    loadTrainingData(filePath_train+"data14.xml");
    loadTrainingData(filePath_train+"data15.xml");
    loadTrainingData(filePath_train+"data16.xml");
    loadTrainingData(filePath_train+"data17.xml");

    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_angry_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_contemptuous_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_disgusted_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_fearful_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_happy_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_neutral_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_sad_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_03_Caucasian_male_surprised_frontal.xml");

    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_angry_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_contemptuous_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_disgusted_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_fearful_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_happy_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_neutral_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_sad_frontal.xml");
    loadTrainingData(filePath_train+"Rafd090_07_Caucasian_male_surprised_frontal.xml");

    //Train aam with Training Data
    //optional: Set number of used Shape/Appearance Parameters
    aam.setNumShapeParameters(numShapeParameters);
    aam.setNumAppParameters(numAppParameters);
    aam.train();

    namedWindow(WINDOW_NAME, WINDOW_AUTOSIZE);

    VideoCapture cam;
    cam.open(filePath_test+"video.webm");

    if(!cam.isOpened()) {
        cout<<"Video konnte nicht geladen werden"<<endl;
        return 0;
    }

    //Load image and initialize the fitting shape
    Mat fittingImage;
    
    cam.read(fittingImage);
    aam.setFittingImage(fittingImage);   //Converts image to right format
    aam.resetShape();    //Uses Viola-Jones Face Detection to initialize shape

    //Terminate until fitting parameters change under predefined threshold
    // or 100 update steps have been executed
    while(cam.read(fittingImage)) {
        aam.setFittingImage(fittingImage);   //Converts image to right format

        //Initialize with value > fitThreshold to enter the fitting loop
        float fittingChange = 20.0f;
        int steps = 0;

        while(fittingChange > fitThreshold) {
            fittingChange = aam.fit();   //Execute single update step
            steps++;
            cout<<"Step "<<steps<<" || Error per pixel: "<<aam.getErrorPerPixel()<<" || Parameter change: "<<fittingChange<<endl;
        }

        Mat image = fittingImage.clone();
        Mat p = aam.getFittingShape();
        image = drawShape(image, p);
        imshow(WINDOW_NAME, image);
        waitKey(1);
    }

    waitKey(0);
    return 0;
}

