#include <bits/stdc++.h>
#include <opencv2/core.hpp>
#include <opencv.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
using namespace std;
using namespace cv;
int label = 0;
double thresh = 0.0;
void seedFilling(const Mat& srcImg, Mat& labelImg)
{
	labelImg.release();
	srcImg.convertTo(labelImg, CV_32SC1);
	//srcImg.convertTo(labelImg, CV_8UC1);
	int row = srcImg.rows;
	int col = srcImg.cols;
	Mat mask(row, col, CV_8UC1);
	mask.setTo(0);
	for (int i = 0; i < row; i++)
	{
		int* data = labelImg.ptr<int>(i);//
		uchar* maskptr = mask.ptr<uchar>(i);//标记数组
		for (int j = 0; j < col; j++)
		{
			if(data[j]==255&&mask.at<uchar>(i,j)!=1)
			{
				mask.at<uchar>(i, j) = 1;
				stack<pair<int, int> > nearPixel;
				nearPixel.push(pair<int, int>(i, j));//将该位置入列
				++label;
				while (!nearPixel.empty())
				{
					pair<int, int> cur_pixel = nearPixel.top();
					int cy = cur_pixel.first;
					int cx = cur_pixel.second;
					labelImg.at<int>(cy, cx) = label;
					nearPixel.pop();
					if (cx - 1 >= 0)
					{
						//left
						if (labelImg.at<int>(cy, cx - 1) == 255 && mask.at<uchar>(cy, cx - 1) != 1)
						{
							nearPixel.push(pair<int, int>(cy, cx - 1));
							mask.at<uchar>(cy, cx - 1) = 1;
						}
						//left&up
						if (cy - 1 >= 0)
						{
							if (labelImg.at<int>(cy - 1, cx - 1) == 255 && mask.at<uchar>(cy - 1, cx - 1) != 1)
							{
								nearPixel.push(pair<int, int>(cy - 1, cx - 1));
								mask.at<uchar>(cy - 1, cx - 1) = 1;
							}
						}
						//left&down
						if (cy + 1 < row)
						{
							if (labelImg.at<int>(cy + 1, cx - 1) == 255 && mask.at<uchar>(cy + 1, cx - 1) != 1)
							{
								nearPixel.push(pair<int, int>(cy + 1, cx - 1));
								mask.at<uchar>(cy + 1, cx - 1) = 1;
							}
						}
					}
					if (cx + 1 < col)
					{
						//right
						if (labelImg.at<int>(cy, cx + 1) == 255 && mask.at<uchar>(cy, cx + 1) != 1)
						{
							nearPixel.push(pair<int, int>(cy, cx + 1));
							mask.at<uchar>(cy, cx + 1) = 1;
						}
						//right&up
						if (cy - 1 >= 0)
						{
							if (labelImg.at<int>(cy - 1, cx + 1) == 255 && mask.at<uchar>(cy - 1, cx + 1) != 1)
							{
								nearPixel.push(pair<int, int>(cy - 1, cx + 1));
								mask.at<uchar>(cy - 1, cx + 1) = 1;
							}
						}
						//right&down
						if (cy + 1 < row)
						{
							if (labelImg.at<int>(cy + 1, cx + 1) == 255 && mask.at<uchar>(cy + 1, cx + 1) != 1)
							{
								nearPixel.push(pair<int, int>(cy + 1, cx + 1));
								mask.at<uchar>(cy + 1, cx + 1);
							}
						}
					}
					//up
					if (cy - 1 >= 0)
					{
						if (labelImg.at<int>(cy - 1, cx) == 255 && mask.at<uchar>(cy - 1, cx) != 1)
						{
							nearPixel.push(pair<int, int>(cy - 1, cx));
							mask.at<uchar>(cy - 1, cx) = 1;
						}
					}
					//down
					if (cy + 1 < row)
					{
						if (labelImg.at<int>(cy + 1, cx) == 255 && mask.at<uchar>(cy + 1, cx) != 1)
						{
							nearPixel.push(pair<int, int>(cy + 1, cx));
							mask.at<uchar>(cy + 1, cx) = 1;
						}
					}
				}
			}
		}
	}
}

void removeSamll(Mat const& srcImg, Mat &dstImg)
{
	dstImg = srcImg.clone();
	vector<vector<Point> > contours;
	vector<Vec4i> h;
	thresh = 0.0;
	double area;
	findContours(srcImg, contours, h, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point());
	if (!contours.empty() && !h.empty())
	{
		vector<vector<Point> >::const_iterator it = contours.begin();
		while (it != contours.end())
		{
			area = contourArea(*it);
			if (area > thresh)
			{
				thresh = area;
			}
			it++;
		}
		it = contours.begin();
		while (it != contours.end())
		{
			Rect rect = boundingRect(Mat(*it));
			area = contourArea(*it);
			if (area < thresh)
			{
				for (int i = rect.y; i < rect.y + rect.height; i++)
				{
					uchar* pixel = dstImg.ptr<uchar>(i);
					for (int j = rect.x; j < rect.x + rect.width; j++)
					{
						if (pixel[j] == 255)
						{
							pixel[j] = 0;
						}
					}
				}
			}
			it++;
		}
	}
}

int main()
{
	Mat img = imread("E:/1ComputerVision/exp/exp5/horse_mask.png");
	if (img.empty())
	{
		cout << "image read failed!";
		return -1;
	}
	Mat imgGray, imgBinary;
	cvtColor(img, imgGray, COLOR_RGB2GRAY);
	threshold(imgGray, imgBinary, 127, 255, THRESH_BINARY);
	imshow("Image", img);
    imshow("ImageBinary", imgBinary);
	Mat temp;
	seedFilling(imgBinary, temp);
	//seedFilling(img, temp);
	cout << "the connected components are : " << label<<"\n";
	Mat temp2;
	removeSamll(imgBinary, temp2);
	imshow("image2", temp2);
	cout << "\n" << "the maximum area is : " << thresh << "\n";
	waitKey(0);
	return 0;
}