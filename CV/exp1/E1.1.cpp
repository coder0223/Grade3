#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/core.hpp>
#include <bits/stdc++.h>
using namespace cv;
using namespace std;

int contr;//�Աȶȱ���
Mat image, temp;//ͼƬ

static void Contrast(int, void*)//�����Աȶȵĺ���
{
	namedWindow("Image", WINDOW_AUTOSIZE);//��������
	for (int y = 0; y < image.rows; y++)
	{
		for (int x = 0; x < image.cols; x++)
		{
			for (int c = 0; c < 3; c++)//����ÿ�����ص�(x,y)ÿ��ͨ��c��ֵ
			{
				double p = ((image.at<Vec3b>(y, x)[c] - 127) / 225.00) * contr * 0.1;//sigmoid����ԭ��Ϊ1/(1+e^(-x))
				temp.at<Vec3b>(y, x)[c] = saturate_cast<uchar>(image.at<Vec3b>(y, x)[c] * ((1.00 / (1.00 + exp(-p))) + 0.3));//sigmiod����
			}	//saturate_cast��ͼ��ɫ�ʽ��б�������ֹ�������
		}
	}
	imshow("Image", image);//չʾ��ʼͼƬ
	imshow("Slider", temp);//չʾ�Աȶȿɵ���ͼƬ
}
int main()
{
	image = imread("E:/1ComputerVision/exp/exp1/exp1.1/pict.jpg");//��ȡͼƬ
	if (image.empty())
	{
		cout << "image can not load!\n";
		return -1;
	}//��ͼƬ�����ڣ����ر�����Ϣ���˳�
	temp = Mat::zeros(image.size(), image.type());//����һ�ź�ԭʼͼƬ��ȫһ�µ�ͼƬ
	contr = 20;//��ʼ�Աȶ�����Ϊ20
	namedWindow("Slider", WINDOW_AUTOSIZE);//����Slider����������
	createTrackbar("Contrast", "Slider", &contr, 150, Contrast);//�����������ĺ���
	//�켣�����ơ��������ơ������ʼλ�á������ܵ�������ֵ��ָ��ص�����
	Contrast(contr, 0);
	waitKey(0);
	return 0;
}