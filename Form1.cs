using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.Data;
using System.Drawing;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows.Forms;
using Emgu.CV.Util;
using Emgu.CV;
using Emgu.CV.UI;
using Emgu.CV.Structure;
using System.Drawing;
using Emgu.CV.CvEnum;
using System.Collections;
using System.IO;


namespace IAProject
{
    public partial class Form1 : Form
    {
        Capture capture; //create a camera captue
        Timer ImageTimer;
        Image<Bgr, Byte> MyImage;
        Image<Gray, Byte> Mask;
        Image<Bgr, Byte> MyImageRegion;
        Image<Gray, Byte> MyMaskRegion;
        private HaarCascade haar;
        Seq<Point> Hull;
        Seq<MCvConvexityDefect> Defects;
        int fingerNum;
        MCvConvexityDefect[] DefectArray;
        Seq<Point> filteredHull;

        public Form1()
        {
            InitializeComponent();
            capture = new Capture(); //create a camera captue
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_WIDTH, 1280);
            capture.SetCaptureProperty(Emgu.CV.CvEnum.CAP_PROP.CV_CAP_PROP_FRAME_HEIGHT, 720);
            ImageTimer = new Timer();
            ImageTimer.Interval = 1000/24;
            ImageTimer.Tick += new EventHandler(ImageTimer_Tick);
            ImageTimer.Start();
            haar = new HaarCascade("C:\\haarcascade_frontalface_default.xml");
            ArrayList Images=new ArrayList();
            /*foreach (string file in Directory.EnumerateFiles("C:\\standard_test_images\\", "*.tif"))
            {
                Image<Bgr, Byte> I = new Image<Bgr, byte>(file);
                ImageTimer_Tick(I);
            }*/

        }
        public void ImageTimer_Tick(object sender, EventArgs e)
        {
            //MyImage = I;
            MyImage= capture.QueryFrame();
            if (MyImage != null)
            {
                Image<Bgr, Byte> MyNoFaceImage = RemoveFace(MyImage);
                imageBox4.Image = MyNoFaceImage;
                Mask = SkinDetect(MyNoFaceImage, new Hsv(0, 10, 60), new Hsv(20, 150, 255), 0);
                imageBox1.Image = MyImage;
                imageBox2.Image = Mask;
                Image<Gray, Byte> CannyImage = new Image<Gray, Byte>(Mask.Size);
                CvInvoke.cvCanny(Mask, CannyImage, 30, 60, 3);
                imageBox3.Image = CannyImage;
                MyMaskRegion = GetMask();
                imageBox5.Image = MyMaskRegion;
                imageBox4.Image = MyImageRegion;
                MyImageRegion.Resize(100, 100, INTER.CV_INTER_CUBIC);
                ExtractFingerNumber(MyMaskRegion);

                imageBox1.SizeMode = PictureBoxSizeMode.Zoom;
                imageBox2.SizeMode = PictureBoxSizeMode.Zoom;
                imageBox3.SizeMode = PictureBoxSizeMode.Zoom;
                imageBox4.SizeMode = PictureBoxSizeMode.Zoom;
                imageBox5.SizeMode = PictureBoxSizeMode.Zoom;
                imageBox6.SizeMode = PictureBoxSizeMode.Zoom;
            }
        }

        
        private Image<Bgr, Byte> RemoveFace(Image<Bgr, Byte> Input)
        {
            Image<Bgr, Byte> Output = Input.Copy();
            Image<Gray, Byte> GrayImage = Input.Convert<Gray, Byte>();
            var faces = GrayImage.DetectHaarCascade(
                haar, 1.4, 4,
                HAAR_DETECTION_TYPE.DO_CANNY_PRUNING,
                new Size(Input.Width / 8, Input.Height / 8))[0];
            foreach (var face in faces)
            {
                Rectangle R = face.rect;
                Ellipse E = new Ellipse(new PointF(R.Location.X + R.Width / 2, R.Location.Y + R.Width / 2), new SizeF(R.Width, (int)(R.Height * 1.2)), 90);
                Output.Draw(E, new Bgr(Color.Black), -1);
            }
            return Output;
        }


        private Image<Gray, Byte> SkinDetect(Image<Bgr, Byte> Input, IColor min, IColor max, int a)
        {
            Image<Gray, byte> skin = new Image<Gray, byte>(Input.Width, Input.Height);
            if (a == 1)
            {
                Image<Ycc, Byte> YCrCbInput = Input.Convert<Ycc, Byte>();
                skin = YCrCbInput.InRange((Ycc)min, (Ycc)max);
                imageBox4.Image = YCrCbInput;
            }
            else
            {
                Image<Hsv, Byte> HsvInput = Input.Convert<Hsv, Byte>();
                skin = HsvInput.InRange((Hsv)min, (Hsv)max);
                imageBox4.Image = HsvInput;
            }
            StructuringElementEx Rect12 = new StructuringElementEx(12, 12, 6, 6, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_CROSS);
            StructuringElementEx Rect6 = new StructuringElementEx(6, 6, 3, 3, Emgu.CV.CvEnum.CV_ELEMENT_SHAPE.CV_SHAPE_CROSS);

            CvInvoke.cvErode(skin, skin, Rect6, 1);
            CvInvoke.cvDilate(skin, skin, Rect12, 2);
            
            return skin;
        }

        private Image<Gray, Byte> GetMask()
        {

            using (MemStorage storage = new MemStorage()) //allocate storage for contour approximation
            {
                Contour<Point> contours = Mask.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage);
                Contour<Point> Big = null;

                Double BigArea = 0;

                while (contours != null)
                {
                    if (contours.Area > BigArea)
                    {
                        BigArea = contours.Area;
                        Big = contours;
                    }
                    contours = contours.HNext;
                }

                if (Big != null)
                {
                    Contour<Point> currentContour = Big.ApproxPoly(Big.Perimeter * 0.0025, storage);
                    MyImage.Draw(currentContour, new Bgr(Color.LimeGreen), 5);
                    var Ract = currentContour.BoundingRectangle;
                    int Diff = Ract.Height - Ract.Width;

                    if (Diff < 0)
                    {
                        Ract.Height += Math.Abs(Diff);
                        Ract.Y -= Math.Abs(Diff) / 2;
                        if (Ract.Y < 0) { Ract.Y = 0; }
                        if (Ract.Y + Ract.Height > MyImage.Height) { Ract.Y = MyImage.Height - Ract.Height; }
                    }
                    else
                    {
                        Ract.Width += Math.Abs(Diff);
                        Ract.X -= Math.Abs(Diff) / 2;
                        if (Ract.X < 0) { Ract.X = 0; }
                        if (Ract.X + Ract.Width > MyImage.Width) { Ract.X = MyImage.Width - Ract.Width; }
                    }
                    Ract.Inflate(new Size(40, 40));
                    MyImageRegion = MyImage.Copy();
                    Image<Gray, Byte> MYROI = Mask.Copy();
                    MyImageRegion.ROI = Ract;
                    MYROI.ROI = Ract;
                    return MYROI;
                }
            }
            return null;
        }

        private void ExtractFingerNumber(Image<Gray, Byte> MyMask)
        {
            using (MemStorage storage = new MemStorage()) //allocate storage for contour approximation
            {
                Contour<Point> contours = MyMask.FindContours(Emgu.CV.CvEnum.CHAIN_APPROX_METHOD.CV_CHAIN_APPROX_SIMPLE, Emgu.CV.CvEnum.RETR_TYPE.CV_RETR_LIST, storage);
                Contour<Point> Big = null;
                Double BigArea = 0;

                Contour<Point> Small = null;
                Double SmallArea = 100000;

                while (contours != null)
                {
                    if (contours.Area > BigArea)
                    {
                        BigArea = contours.Area;
                        Big = contours;
                    }
                    if ((contours.Area < SmallArea)&&(contours.Area>40))
                    {
                        SmallArea = contours.Area;
                        Small = contours;
                    }
                    contours = contours.HNext;
                }


                CircleF This = new CircleF() ;

                if (Big != null)
                {
                    Contour<Point> currentContour = Big.ApproxPoly(Big.Perimeter * 0.0025, storage);
                    MyImageRegion.Draw(currentContour, new Bgr(Color.LimeGreen), 5);

                    Hull = Big.GetConvexHull(Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);
                    Defects = Big.GetConvexityDefacts(storage, Emgu.CV.CvEnum.ORIENTATION.CV_CLOCKWISE);

                    DefectArray = Defects.ToArray();
                    MCvBox2D box = Big.GetMinAreaRect();

                    MyImageRegion.DrawPolyline(Hull.ToArray(), true, new Bgr(200, 125, 75), 2);
                    //MyImage.DrawPolyline(Defects.ToArray(), true, new Bgr(255, 255, 0), 2);

                    List<CircleF> SmallPoints = new List<CircleF>();
                    List<CircleF> BigPoints=new List<CircleF>();
                    List<LineSegment2D> FingerLines=new List<LineSegment2D>();

                    Double angled=0;
                    fingerNum = 0;
                    for (int i = 0; i < DefectArray.Length; i++)
                    {
                        PointF startPoint = new PointF((float)DefectArray[i].StartPoint.X, (float)DefectArray[i].StartPoint.Y);
                        PointF depthPoint = new PointF((float)DefectArray[i].DepthPoint.X, (float)DefectArray[i].DepthPoint.Y);
                        PointF endPoint = new PointF((float)DefectArray[i].EndPoint.X, (float)DefectArray[i].EndPoint.Y);

                        LineSegment2D startDepthLine = new LineSegment2D(DefectArray[i].StartPoint, DefectArray[i].DepthPoint);
                        LineSegment2D depthEndLine = new LineSegment2D(DefectArray[i].DepthPoint, DefectArray[i].EndPoint);

                        
                        CircleF startCircle = new CircleF(startPoint, 5f);
                        CircleF depthCircle = new CircleF(depthPoint, 5f);
                        CircleF endCircle = new CircleF(endPoint, 5f);
                        

                        if ((startCircle.Center.Y < box.center.Y || depthCircle.Center.Y < box.center.Y) && (startCircle.Center.Y < depthCircle.Center.Y) && (Math.Sqrt(Math.Pow(startCircle.Center.X - depthCircle.Center.X, 2) + Math.Pow(startCircle.Center.Y - depthCircle.Center.Y, 2)) > box.size.Height / 7))
                        {
                            fingerNum++;
                            MyImageRegion.Draw(startDepthLine, new Bgr(Color.Black), 2);
                            SmallPoints.Add(depthCircle);
                            BigPoints.Add(startCircle);
                            FingerLines.Add(startDepthLine);
                        }
                    }

                    foreach (CircleF a in SmallPoints)
                    {
                        MyImageRegion.Draw(a, new Bgr(Color.Yellow), 5);
                    }

                    foreach (CircleF a in BigPoints)
                    {
                        MyImageRegion.Draw(a, new Bgr(Color.Red), 5);
                    }

                    MyImageRegion.Draw(This, new Bgr(Color.Pink), 15);
                    label2.Text="";
                    if (fingerNum == 5)
                    {
                        label2.Text = "Open Palm";
                    }

                    if (fingerNum == 0)
                    {
                        if (box.size.Height / box.size.Width > 1.5)
                        {
                            label2.Text = "closed Palm";
                        }
                        else
                        {
                            label2.Text = "Fist";
                        }
                    }
                    if (fingerNum == 3)
                    {
                        if (BigArea/SmallArea<20)
                        {
                            label2.Text = "3 Fingers";
                        }
                    }

                    if (fingerNum == 2)
                    {
                        if((Math.Abs(FingerLines[0].GetExteriorAngleDegree(FingerLines[1]))<100)&&(Math.Abs(FingerLines[0].GetExteriorAngleDegree(FingerLines[1]))>70))
                        {
                            label2.Text = "Gun Sign";
                        }
                        if ((Math.Abs(FingerLines[0].GetExteriorAngleDegree(FingerLines[1])) < 50) && (Math.Abs(FingerLines[0].GetExteriorAngleDegree(FingerLines[1])) > 20))
                        {
                            label2.Text = "Victory";
                        }
                    }

                    label1.Text = fingerNum.ToString();


                }
            }
        }

        private void imageBox1_Click(object sender, EventArgs e)
        {
        }

        private void imageBox4_Click(object sender, EventArgs e)
        {

        }

        private void label1_Click(object sender, EventArgs e)
        {

        }

        private void label2_Click(object sender, EventArgs e)
        {

        }

        private void imageBox6_Click(object sender, EventArgs e)
        {

        }
    }
}
