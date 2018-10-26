//#include <QCoreApplication>

//int main(int argc, char *argv[])
//{
//    QCoreApplication a(argc, argv);

//    return a.exec();
//}

#define FILENAME "SheppLogan.png"

#include <fstream> // file I/O
#include <cmath>
#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>

using namespace cv;
using namespace std;

void padImage(const Mat &src, Mat &dst){
    int h = src.size().height;
    int w = src.size().width;
    int diagLen = sqrt(pow(w, 2) + pow(h, 2));
    int padsize1 = ceil((diagLen - h)/2);
    int padsize2 = ceil((diagLen - w)/2);
    copyMakeBorder(src, dst, padsize1, padsize1, padsize2, padsize2, BORDER_CONSTANT, Scalar(255));
}


int main(){
    string s ("/home/clay/Documents/code/projects/filt-back-proj/cpp/filtbackproj/SheppLogan.png");
    Mat image = imread(s, IMREAD_GRAYSCALE);
    string window_name = "yo";
    //namedWindow(window_name, CV_WINDOW_AUTOSIZE);
    //imshow( "Display window", image);

    // pad image
    Mat image_pad;
    padImage(image, image_pad);

    Point center = Point(image_pad.cols/2, image_pad.rows/2);
    double angle = 45.0;
    double scale = 1.0;
    Mat rot_mat = getRotationMatrix2D(center, angle, scale);
    Mat image_rot;
    warpAffine(image_pad, image_rot, rot_mat, image_pad.size());

    imshow("Display window", image_rot);


    waitKey(0);
    return 0;


}
