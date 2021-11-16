#include <opencv2\xfeatures2d.hpp>
#include <bits/stdc++.h>
#include <opencv.hpp>
#include <opencv2/features2d.hpp>

using namespace std;
using namespace cv;

int main()
{
	//double time0 = static_cast<double>(getTickCount());
	Mat imageL = cv::imread("E:/1ComputerVision/exp/exp8/1.jpg");
	Mat imageR = cv::imread("E:/1ComputerVision/exp/exp8/3.jpg");
	//imshow("imageL", imageL);
	//imshow("imageR", imageR);
	//Ptr<cv::xfeatures2d::SIFT> sift = cv::xfeatures2d::SIFT::create();
	Ptr<cv::xfeatures2d::SURF> surf = cv::xfeatures2d::SURF::create();
	//Ptr<cv::ORB> orb = cv::ORB::create();
	vector<cv::KeyPoint> kpL, kpR;
	//单独提取特征点
	//sift->detect(imageL, kpL);
	//sift->detect(imageR, kpR);
	surf->detect(imageL, kpL);
	surf->detect(imageR, kpR);
	//orb->detect(imageL, kpL);
	//orb->detect(imageR, kpR);
	//画特征点
	Mat kpImageL;
	Mat kpImageR;
	drawKeypoints(imageL, kpL, kpImageL, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);
	drawKeypoints(imageR, kpR, kpImageR, cv::Scalar::all(-1), cv::DrawMatchesFlags::DRAW_RICH_KEYPOINTS);

	//特征点匹配
	Mat despL, despR;
	//提取特征点并计算特征描述子
	//sift->detectAndCompute(imageL, cv::Mat(), kpL, despL);
	//sift->detectAndCompute(imageR, cv::Mat(), kpR, despR);
	surf->detectAndCompute(imageL, cv::Mat(), kpL, despL);
	surf->detectAndCompute(imageR, cv::Mat(), kpR, despR);
	//orb->detectAndCompute(imageL, cv::Mat(), kpL, despL);
	//orb->detectAndCompute(imageR, cv::Mat(), kpR, despR);
	std::vector<cv::DMatch> matches;
	if (despL.type() != CV_32F || despR.type() != CV_32F)
	{
		despL.convertTo(despL, CV_32F);
		despR.convertTo(despR, CV_32F);
	}
	Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("FlannBased");
	matcher->match(despL, despR, matches);
	//计算特征点距离的最大值 
	double maxDist = 0;
	for (int i = 0; i < despL.rows; i++)
	{
		double dist = matches[i].distance;
		if (dist > maxDist)
			maxDist = dist;
	}
	//挑选好的匹配点
	std::vector< DMatch > good_matches;
	for (int i = 0; i < despL.rows; i++)
	{
		if (matches[i].distance < 0.5 * maxDist)
		{
			good_matches.push_back(matches[i]);
		}
	}
	//显示特征点
	imshow("KPL", kpImageL);
	imshow("KPR", kpImageR);
	//显示匹配效果
	Mat imageOutput;
	drawMatches(imageL, kpL, imageR, kpR, good_matches, imageOutput);
	imshow("matching", imageOutput);
	waitKey(0);	
	//time0 = ((double)getTickCount() - time0) / getTickFrequency();
	//cout << "runtime ：" << time0 << "s";
	return 0;
}
