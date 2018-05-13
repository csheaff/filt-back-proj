//http://cimg.eu/reference/group__cimg__tutorial.html

#include "CImg.h"
#include <fstream>
#include <iostream>

using namespace cimg_library;
using namespace std;

int main() {
  //std::ifstream fin("/home/csheaff/code/filt-back-proj");  //include a new path

  CImg<float> img("../../SheppLogan.png"); //3 channel
  img.channel(0); //now single channel

  CImgDisplay main_disp(img,"My Image"); //, draw_disp(img,"My Image");

  cout<<img<<endl;
  return 0;
}


