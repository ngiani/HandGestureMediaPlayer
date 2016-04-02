﻿using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV;
using Emgu.CV.Util;
using Emgu.CV.Cvb;
using Emgu.CV.Structure;
using Emgu.CV.Shape;

namespace HandGestureMediaPlayer
{
    public partial class Form1 : Form
    {

        bool capturing;
        int minH, minS, minV;
        int maxH, maxS, maxV;

        int erosions, dilations;

        Point[] handContour;
        Dictionary<int,List<Point[]>> trainingContours;

        bool enableClassification;

        int captureCounter;

        Capture camera;

        public Form1()
        {
            InitializeComponent();

            minH = minS  = minV = 0;

            maxH = maxS = maxV = 0;

            erosions = dilations = 0;

            enableClassification = false;

            trainingContours = new Dictionary<int, List<Point[]>>();


            captureCounter = 0;


            try
            {
                camera = new Capture();
            }

            catch (TypeInitializationException exc)
            {
                MessageBox.Show(exc.Message);
            }


        }

        private void trackBar2_Scroll(object sender, EventArgs e)
        {
            minS = trackBar2.Value;
            label2.Text = minS.ToString();
        }

        private void trackBar3_Scroll(object sender, EventArgs e)
        {
            minV = trackBar3.Value;
            label3.Text = minV.ToString();
        }

        private void trackBar4_Scroll(object sender, EventArgs e)
        {
            maxH = trackBar4.Value;
            label4.Text = maxH.ToString();
        }

        private void trackBar5_Scroll(object sender, EventArgs e)
        {
            maxS = trackBar5.Value;
            label5.Text = maxS.ToString();
        }

        private void trackBar6_Scroll(object sender, EventArgs e)
        {
            maxV = trackBar6.Value;
            label6.Text = maxV.ToString();
        }

        private void trackBar8_Scroll(object sender, EventArgs e)
        {
            dilations = trackBar8.Value;
            label8.Text = dilations.ToString();
        }

        private void button2_Click(object sender, EventArgs e)
        {
            if (!trainingContours.ContainsKey(0))
                trainingContours.Add(0, new List<Point[]>());

            trainingContours[0].Add(handContour);
        }

        private void button3_Click(object sender, EventArgs e)
        {
            if (!trainingContours.ContainsKey(1))
                trainingContours.Add(1, new List<Point[]>());

            trainingContours[1].Add(handContour);
        }

        private void trackBar7_Scroll(object sender, EventArgs e)
        {
            erosions = trackBar7.Value;
            label7.Text = erosions.ToString();
        }

      

        private void Form1_Load(object sender, EventArgs e)
        {
            capturing = false;
            
        }

        private void openToolStripMenuItem_Click(object sender, EventArgs e)
        {

            openFileDialog1.Title = "Select video file..";

            openFileDialog1.Filter = "Media Files|*.mpg;*.avi;*.wma;*.mov;*.wav;*.mp2;*.mp3|All Files|*.*";

            try {
                if (openFileDialog1.ShowDialog() == System.Windows.Forms.DialogResult.OK) {
                    axWindowsMediaPlayer1.URL = (openFileDialog1.FileName);

                    
                }
            }
            catch (ArgumentException ex) {
                MessageBox.Show(ex.Message.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
            catch (Exception ex) {
                MessageBox.Show(ex.Message.ToString(), "Error", MessageBoxButtons.OK, MessageBoxIcon.Error);
            }
        }

        private double contoursDistance(Point[] contour1, Point[] contour2)
        {

            HausdorffDistanceExtractor HDExtractor = new HausdorffDistanceExtractor();


            return HDExtractor.ComputeDistance(contour1, contour2);

        }


        private int NNContours(Dictionary<int,List<Point[]>> trainingSet, Point[] test)
        {

            int nearestLabel = 0;
            double minDistance = 100000;

            foreach (int key in trainingSet.Keys)
            {
                foreach (Point[] contour in trainingSet[key])
                {
                    double currentDistance = contoursDistance(contour, test);

                    if (currentDistance < minDistance)
                    {
                        minDistance = currentDistance;
                        nearestLabel = key;
                    }
                }
            }


            return nearestLabel;

        }


        private void ProcessFrame(object sender, EventArgs args)
        {

            captureCounter++;


            Mat frame = camera.QueryFrame();

           

            Image<Bgr, Byte> img = frame.ToImage<Bgr, Byte>();

            Image<Hsv, Byte> HSVimg = img.Convert<Hsv,Byte>();


            Image<Gray, Byte> binary = HSVimg.InRange(new Hsv(minH, minS, minV), new Hsv(maxH, maxS, maxV));

            Image<Gray, Byte> eroded = binary.Erode(erosions);

            Image<Gray, Byte> dilated = eroded.Dilate(dilations);

            CvBlobDetector blobDetector = new CvBlobDetector();
            CvBlobs blobs = new CvBlobs();

            blobDetector.Detect(dilated, blobs);


            int maxBlobArea = 0;
            CvBlob largestBlob = null;

           foreach (CvBlob blob in blobs.Values)
            {
                if (blob.Area > maxBlobArea) {
                    maxBlobArea = blob.Area;
                    largestBlob = blob;
                }
            }




            if (largestBlob != null && largestBlob.Area >= 10000)
            {
                handContour = largestBlob.GetContour();

                VectorOfInt convexHullIndices = new VectorOfInt();

                VectorOfPoint convexHull = new VectorOfPoint();

                CvInvoke.ConvexHull(new VectorOfPoint(handContour), convexHull);

                CvInvoke.ConvexHull(new VectorOfPoint(handContour), convexHullIndices);

                Mat defects = new Mat();


                img.Draw(handContour, new Bgr(0, 0, 255));
                img.Draw(convexHull.ToArray(), new Bgr(255, 0, 0), 2);


                try
                {
                    CvInvoke.ConvexityDefects(new VectorOfPoint(handContour), convexHullIndices, defects);

                }

                catch(CvException exc)
                {
                    MessageBox.Show(exc.Message);
                }

                if (!defects.IsEmpty) { 

                    Matrix<int> defectsInt = new Matrix<int>(defects.Rows, defects.Cols, defects.NumberOfChannels);

                    defects.CopyTo(defectsInt);


                    float sumOfDistances = 0;

                    for (int i = 0; i < defectsInt.Rows; i++)
                    {
                        int startIdx = defectsInt.Data[i, 0];
                        int endIdx = defectsInt.Data[i, 1];
                        int farthestIdx = defectsInt.Data[i, 2];
                        float distance = defectsInt.Data[i, 3];

                        sumOfDistances += distance;

                        Point farthestPoint = handContour[farthestIdx];

                        img.Draw(new CircleF(farthestPoint, 2.0f), new Bgr(0, 255, 0), -1);
                    }

                    float meanDistance = (sumOfDistances / defectsInt.Rows);

                    if (meanDistance >= 5000)
                    {
                        label10.Text = "Pause";
                        try {
                            axWindowsMediaPlayer1.Ctlcontrols.pause();
                        }

                        catch (Exception exc)
                        {
                            MessageBox.Show(exc.Message);
                        }
                    }

                    if (meanDistance < 4000)
                    {
                        label10.Text = "Play";
                        try
                        {
                            axWindowsMediaPlayer1.Ctlcontrols.play();
                        }

                        catch (Exception exc)
                        {
                            MessageBox.Show(exc.Message);
                        }
                    }
              

                }

            }

            pictureBox1.Image = img.Bitmap;
        }


        private void button1_Click(object sender, EventArgs e)
        {


           
            if (camera!=null)
            {

                if (capturing == true)
                {

  

                    capturing = false;
                    button1.Text = "Start Capture";
                    Application.Idle -= ProcessFrame;
                }

                else
                {



                    capturing = true;
                    button1.Text = "Stop Capture";

           

                    Application.Idle += ProcessFrame;
                }


            }
        }

        private void trackBar1_Scroll(object sender, EventArgs e)
        {
            minH = trackBar1.Value;

            label1.Text = minH.ToString();
        }
    }
}