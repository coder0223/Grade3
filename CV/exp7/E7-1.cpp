#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv.hpp>
#include <bits/stdc++.h>
using namespace std;
using namespace cv;

int main(int argc, char** argv)
{
	double time0 = static_cast<double>(getTickCount());
	Mat img = imread("E:/1ComputerVision/exp/exp7/pic3.jpg");
	if (img.empty())
	{
		cout << "image read failed!" << endl;
		return -1;
	}
	Mat image;
	cvtColor(img, image, COLOR_RGB2GRAY);
	int rows = image.rows - 1, cols = image.cols - 1;//行列数
	int row, col, i, j;
	int Ix, Iy;
	Mat M(rows + 1, cols + 1, CV_32FC3, Scalar::all(0));	//构建M矩阵:三个通道的int类型的

	int sideAdd, sideDes;
	for (row = 0; row <= rows; row++)
	{
		for (col = 0; col <= cols; col++)
		{
			//step1: 计算方向梯度
			sideAdd = 1, sideDes = 1;
			if (row == 0 || col == 0)	sideDes = 0;
			if (row == rows || col == cols)		sideAdd = 0;
			Ix = (image.at<uchar>(row + sideAdd, col) - image.at<uchar>(row - sideDes, col));
			Iy = (image.at<uchar>(row, col + sideAdd) - image.at<uchar>(row, col - sideDes));

			//step2: 计算两个方向梯度乘积
			M.at<Vec3f>(row, col)[0] = pow(Ix, 2);
			M.at<Vec3f>(row, col)[1] = Ix * Iy;
			M.at<Vec3f>(row, col)[2] = pow(Iy, 2);
		}
	}
	//step3: 利用高斯函数进行高斯加权
	Mat M_Gauss(rows + 1, cols + 1, CV_32FC3, Scalar::all(0));
	cv::GaussianBlur(M, M_Gauss, cv::Size(5, 5), 3, 3);

	//step4: 每个像素的Harris响应值R，并对小于某一阈值t的R置为零 
	Mat R(rows + 1, cols + 1, CV_32FC1, Scalar::all(0));
	float thresh = 1000000, temp = 0;	//阈值大小
	float detM = 0, traceM = 0;		//行列式和迹
	for (row = 0; row <= rows; row++)
	{
		for (col = 0; col <= cols; col++)
		{
			//计算行列式和迹
			detM = M_Gauss.at<Vec3f>(row, col)[0] * M_Gauss.at<Vec3f>(row, col)[2] - pow(M_Gauss.at<Vec3f>(row, col)[1], 2);
			traceM = M_Gauss.at<Vec3f>(row, col)[0] + M_Gauss.at<Vec3f>(row, col)[2];
			//计算图像每个像元的响应值
			temp = detM - 0.05 * pow(traceM, 2);
			if (temp > thresh)//如果大于规定的阈值
				R.at<float>(row, col) = temp;
		}
	}

	//step5：在3×3或5×5的邻域内进行非最大值抑制，局部最大值点即为图像中的角点。
	float winMax = 0;
	int maxR, maxC;		//窗口内最大值的行列值
	int windowR, windowC;	//窗口内的移动位置
	for (row = 0; row <= rows - 7; row += 7)//窗口大小为7
	{
		for (col = 0; col <= cols - 7; col += 7)
		{
			maxR = 0, maxC = 0;
			winMax = 0;
			for (windowR = 0; windowR < 7; windowR++)
			{
				for (windowC = 0; windowC < 7; windowC++)
				{
					if (winMax < R.at<float>(row + windowR, col + windowC))
					{
						winMax = R.at<float>(row + windowR, col + windowC);
						maxR = row + windowR;
						maxC = col + windowC;
					}
				}
			}
			//画图
			if (maxR > 0 && maxC > 0)
			{
				circle(image, Point(maxC, maxR), 2, Scalar(50, 50, 150), 2, 8, 0);
				//std::cout << maxR << "   " << maxC << ":   " << winMax << endl;
			}
		}
	}
	time0 = ((double)getTickCount() - time0) / getTickFrequency();
	cout << "runtime ：" << time0 << "s" << endl;
	imshow("image", image);
	cv::waitKey(0);
	return 0;
}