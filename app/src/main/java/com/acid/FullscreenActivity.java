/*
 * Copyright (C) 2010,2011,2012 Samuel Audet
 *
 * FacePreview - A fusion of OpenCV's facedetect and Android's CameraPreview samples,
 *               with JavaCV + JavaCPP as the glue in between.
 *
 * This file was based on CameraPreview.java that came with the Samples for 
 * Android SDK API 8, revision 1 and contained the following copyright notice:
 *
 * Copyright (C) 2007 The Android Open Source Project
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 *
 *
 * IMPORTANT - Make sure the AndroidManifest.xml file looks like this:
 *
 * <?xml version="1.0" encoding="utf-8"?>
 * <manifest xmlns:android="http://schemas.android.com/apk/res/android"
 *     package="com.googlecode.javacv.facepreview"
 *     android:versionCode="1"
 *     android:versionName="1.0" >
 *     <uses-sdk android:minSdkVersion="4" />
 *     <uses-permission android:name="android.permission.CAMERA" />
 *     <uses-feature android:name="android.hardware.camera" />
 *     <application android:label="@string/app_name">
 *         <activity
 *             android:name="FacePreview"
 *             android:label="@string/app_name"
 *             android:screenOrientation="landscape">
 *             <intent-filter>
 *                 <action android:name="android.intent.action.MAIN" />
 *                 <category android:name="android.intent.category.LAUNCHER" />
 *             </intent-filter>
 *         </activity>
 *     </application>
 * </manifest>
 */

package com.acid;

import java.io.IOException;
import java.util.List;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.graphics.*;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.media.audiofx.Visualizer;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;

import boofcv.abst.segmentation.ImageSuperpixels;
import boofcv.alg.feature.detect.edge.CannyEdge;
import boofcv.alg.feature.detect.edge.EdgeContour;
import boofcv.alg.feature.detect.edge.EdgeSegment;
import boofcv.alg.filter.binary.BinaryImageOps;
import boofcv.alg.filter.binary.Contour;
import boofcv.alg.filter.binary.impl.BinaryThinning;
import boofcv.alg.filter.blur.GBlurImageOps;
import boofcv.alg.misc.GPixelMath;
import boofcv.alg.misc.ImageMiscOps;
import boofcv.alg.segmentation.ComputeRegionMeanColor;
import boofcv.alg.segmentation.ImageSegmentationOps;
import boofcv.android.ConvertBitmap;
import boofcv.android.VisualizeImageData;
import boofcv.core.encoding.ConvertNV21;
import boofcv.factory.feature.detect.edge.FactoryEdgeDetectors;
import boofcv.factory.segmentation.ConfigFh04;
import boofcv.factory.segmentation.FactoryImageSegmentation;
import boofcv.factory.segmentation.FactorySegmentationAlg;
import boofcv.struct.ConnectRule;
import boofcv.struct.feature.ColorQueue_F32;
import boofcv.struct.image.*;
import com.jabistudio.androidjhlabs.filter.*;
import com.jabistudio.androidjhlabs.filter.util.*;
import georegression.struct.point.Point2D_I32;
import org.ddogleg.struct.FastQueue;
import org.ddogleg.struct.GrowQueue_I32;
//import org.bytedeco.javacpp.opencv_core;
//import org.bytedeco.javacpp.opencv_core.IplImage;
//import org.bytedeco.javacv.AndroidFrameConverter;
//import org.bytedeco.javacv.OpenCVFrameConverter;
//
//import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_32F;
//import static org.bytedeco.javacpp.opencv_core.IPL_DEPTH_8U;
//import static org.bytedeco.javacpp.opencv_core.cvCopy;

// ----------------------------------------------------------------------

public class FullscreenActivity extends Activity {
    private FrameLayout layout;
    private FaceView faceView;
    private Preview mPreview;
    public static int width;
    public static int height;

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        // Hide the window title.
        requestWindowFeature(Window.FEATURE_NO_TITLE);

        super.onCreate(savedInstanceState);
        DisplayMetrics metrics = this.getResources().getDisplayMetrics();
        width = metrics.widthPixels;
        height = metrics.heightPixels;
        getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);

        // Create our Preview view and set it as the content of our activity.
        try {
            layout = new FrameLayout(this);
            faceView = new FaceView(this);
            mPreview = new Preview(this, faceView);
            layout.addView(mPreview);
            layout.addView(faceView);
            setContentView(layout);
        } catch (IOException e) {
            e.printStackTrace();
            new AlertDialog.Builder(this).setMessage(e.getMessage()).create().show();
        }
    }
}

// ----------------------------------------------------------------------

class FaceView extends View implements Camera.PreviewCallback {


    int[] _temp = null;

    int EDGES_THRESHOLD = 70;
    int LAPLACIAN_FILTER_SIZE = 5;
    int MEDIAN_BLUR_FILTER_SIZE = 7;
    int repetitions = 7; // Repetitions for strong cartoon effect.
    int ksize = 1; // Filter size. Has a large effect on speed.
    double sigmaColor = 9; // Filter color strength.
    double sigmaSpace = 7; // Spatial strength. Affects speed.
    int NUM_COLORS = 16;
    int gg = (256 / NUM_COLORS);

    SwimFilter sf = new SwimFilter();
    SwimFilter sf1 = new SwimFilter();
    LaplaceFilter lf = new LaplaceFilter();
    GrayscaleFilter gf = new GrayscaleFilter();
    PosterizeFilter glf = new PosterizeFilter();
    InvertFilter iv = new InvertFilter();

    private float t1;
    private float t2;
    private Bitmap cameraImageBitmap;
    private Bitmap renderImageBitmap;

    //private IplImage renderImageI;
    //private IplImage cameraImageI;

    public FaceView(FullscreenActivity context) throws IOException {
        super(context);

        // Preload the opencv_objdetect module to work around a known bug.
        sf.setAmount(20f);
        sf.setTurbulence(1f);
        sf.setEdgeAction(TransformFilter.CLAMP);
        sf1.setEdgeAction(TransformFilter.CLAMP);
//        sf.setEdgeAction(TransformFilter.RGB_CLAMP);
//        sf1.setEdgeAction(TransformFilter.RGB_CLAMP);

        sf1.setAmount(30f);
        sf1.setTurbulence(1f);
        sf1.setScale(100);
        sf1.setStretch(10);
        glf.setNumLevels(16);
    }

    public void onPreviewFrame(final byte[] data, final Camera camera) {
        try {
            Camera.Size size = camera.getParameters().getPreviewSize();
            processImage(data, size.width, size.height);
            camera.addCallbackBuffer(data);
        } catch (RuntimeException e) {
            // The camera has probably just been released, ignore.
        }
    }

    /**
     * Segments and visualizes the image
     */
    public <T extends ImageBase> void performSegmentation(ImageSuperpixels<T> alg, T color) {
        // Segmentation often works better after blurring the image.  Reduces high frequency image components which
        // can cause over segmentation
        GBlurImageOps.gaussian(color, color, 0.5, -1, null);

        // Storage for segmented image.  Each pixel will be assigned a label from 0 to N-1, where N is the number
        // of segments in the image
        GrayS32 pixelToSegment = new GrayS32(color.width, color.height);

        // Segmentation magic happens here
        alg.segment(color, pixelToSegment);

        // Displays the results
        visualize(pixelToSegment, color, alg.getTotalSuperpixels());
    }

    /**
     * Visualizes results three ways.  1) Colorized segmented image where each region is given a random color.
     * 2) Each pixel is assigned the mean color through out the region. 3) Black pixels represent the border
     * between regions.
     */
    public <T extends ImageBase>
    void visualize(GrayS32 pixelToRegion, T color, int numSegments) {
        // Computes the mean color inside each region
        ImageType<T> type = color.getImageType();
        ComputeRegionMeanColor<T> colorize = FactorySegmentationAlg.regionMeanColor(type);

        FastQueue<float[]> segmentColor = new ColorQueue_F32(type.getNumBands());
        segmentColor.resize(numSegments);

        GrowQueue_I32 regionMemberCount = new GrowQueue_I32();
        regionMemberCount.resize(numSegments);

        ImageSegmentationOps.countRegionPixels(pixelToRegion, numSegments, regionMemberCount.data);
        colorize.process(color, pixelToRegion, regionMemberCount, segmentColor);


        Bitmap outColor = Bitmap.createBitmap(pixelToRegion.width, pixelToRegion.height, Bitmap.Config.RGB_565);
//        Bitmap outSegments = Bitmap.createBitmap(pixelToRegion.width, pixelToRegion.height, Bitmap.Config.ARGB_8888);
//        Bitmap outBorder = Bitmap.createBitmap(pixelToRegion.width, pixelToRegion.height, Bitmap.Config.ARGB_8888);
//
//         Draw each region using their average color
        VisualizeImageData.regionsColor(pixelToRegion, segmentColor, outColor, null);
        //         Draw each region by assigning it a random color
//        VisualizeImageData.regionBorders(pixelToRegion, numSegments, outSegments, null);
//        BufferedImage outSegments = VisualizeRegions.regions(pixelToRegion, numSegments, null);

        // Make region edges appear red

        VisualizeImageData.regionBorders(pixelToRegion, 0xFFFFFF, outColor, null);
//renderImageBitmap=outColor;
//        Bitmap outColorBitmap = Bitmap.createBitmap(outColor.getWidth(), outColor.getHeight(), Bitmap.Config.ARGB_8888);
//        Planar<GrayU8> outColorBoof = new Planar<GrayU8>(GrayU8.class, outColor.getWidth(), outColor.getHeight(), 3);
//        GrayU8 outBorderBoof = new GrayU8(outColor.getWidth(), outColor.getHeight());
//        Planar<GrayU8> out = new Planar<GrayU8>(GrayU8.class, outColor.getWidth(), outColor.getHeight(), 3);
//
//        ConvertBitmap.bitmapToBoof(outColor, outColorBoof, null);
//        ConvertBitmap.bitmapToBoof(outBorder, outBorderBoof, null);
//
        //BinaryImageOps.invert(outBorderBoof, outBorderBoof);
//        BinaryImageOps.logicOr(outColorBoof.getBand(0),outBorderBoof,  out.getBand(0));
//        BinaryImageOps.logicOr(outColorBoof.getBand(1),outBorderBoof,  out.getBand(1));
//        BinaryImageOps.logicOr(outColorBoof.getBand(2), outBorderBoof, out.getBand(2));
//
//        Bitmap renderImage = Bitmap.createBitmap(outColor.getWidth(), outColor.getHeight(), Bitmap.Config.ARGB_8888);
//        ConvertBitmap.boofToBitmap(out, renderImage, null);
//        renderImageBitmap = renderImage;
//        BufferedImage outBorder = new BufferedImage(color.width,color.height,BufferedImage.TYPE_INT_RGB);
        //ConvertBufferedImage.convertTo(color, outBorder, true);
        //VisualizeRegions.regionBorders(pixelToRegion,0xFF0000,outBorder);

        // Show the visualization results
        //ListDisplayPanel gui = new ListDisplayPanel();
        //gui.addImage(outColor,"Color of Segments");
        //  gui.addImage(outBorder, "Region Borders");
        // gui.addImage(outSegments, "Regions");
        //  ShowImages.showWindow(gui,"Superpixels", true);
    }

    protected void processImage(byte[] data, int width, int height) {
        Planar<GrayU8> cameraImageBoof = new Planar<GrayU8>(GrayU8.class, width, height, 3);
        ConvertNV21.nv21ToMsRgb_U8(data, width, height, cameraImageBoof);
        cameraImageBitmap = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
        ConvertBitmap.boofToBitmap(cameraImageBoof, cameraImageBitmap, null);
        Bitmap swimImageBitmap = render(render(cameraImageBitmap, sf), sf1);
//        Planar<GrayU8> swimImageBoof = new Planar<GrayU8>(GrayU8.class, width, height, 3);
//        ConvertBitmap.bitmapToBoof(swimImageBitmap, swimImageBoof, null);
        renderImageBitmap=swimImageBitmap;
//---------------------------------------------------------------------------------------------
        // Select input image type.  Some algorithms behave different depending on image type
//        ImageType<Planar<GrayU8>> imageType = ImageType.pl(3, GrayU8.class);
//		ImageType<Planar<GrayU8>> imageType = ImageType.pl(3,GrayU8.class);
//		ImageType<GrayU8> imageType = ImageType.single(GrayU8.class);
//		ImageType<GrayU8> imageType = ImageType.single(GrayU8.class);

//		ImageSuperpixels alg = FactoryImageSegmentation.meanShift(null, imageType);
//		ImageSuperpixels alg = FactoryImageSegmentation.slic(new ConfigSlic(400), imageType);
//        ImageSuperpixels alg = FactoryImageSegmentation.fh04(new ConfigFh04(100, 30), imageType);
//		ImageSuperpixels alg = FactoryImageSegmentation.watershed(null,imageType);
//        performSegmentation(alg, swimImageBoof);
//
//
        //------------------------------------------------------------------------------------
       /* GrayU8 gray = new GrayU8( width,height);
        GrayU8 edgeImage = gray.createSameShape();
        // creates a gray scale image by averaging intensity value across pixels
        GPixelMath.averageBand(swimImageBoof, gray);
        CannyEdge<GrayU8,GrayS16> canny = FactoryEdgeDetectors.canny(2,true, true, GrayU8.class, GrayS16.class);
        canny.process(gray,0.3f,0.1f,edgeImage);
        List<EdgeContour> edges= canny.getContours();
        List<Contour> contours = BinaryImageOps.contour(edgeImage, ConnectRule.EIGHT, null);
        for( int i = 0; i < contours.size(); i++ ) {
            Contour e = contours.get(i);
                for( Point2D_I32 p : e.external ) {
                    swimImageBoof.getBand(0).set(p.x,p.y,0x00);
                    swimImageBoof.getBand(1).set(p.x,p.y,0x00);
                    swimImageBoof.getBand(2).set(p.x,p.y,0x00);
            }
        }
        Bitmap bm=Bitmap.createBitmap(width,height, Bitmap.Config.ARGB_8888);
        ConvertBitmap.boofToBitmap(swimImageBoof,bm,null);
        renderImageBitmap=bm.copy(Bitmap.Config.ARGB_8888,true);*/
//-------------------------------------------------------------------------------------
//        System.out.println("-------------1");
//        IplImage swimImageI = bitmapToIplImage(swimImageB);
//        System.out.println("-------------2");
//        renderImageBitmap = iplImageToBitmap(swimImageI);
//        System.out.println("-------------3");
//        renderImageBitmap = swimImageB;
        //renderImageI = bitmapToIplImage(cameraImageBitmap);

//        renderImageBitmap=swimImageBitmap;

        //renderImage=render(renderImage,glf);
//        renderImage=render(renderImage,new GrayscaleFilter());

//        IplImage swimImage8u = IplImage.create(swimImage32f.cvSize(), IPL_DEPTH_8U, swimImage32f.nChannels());
//        convertScale(swimImage32f, swimImage8u);
//        System.out.println("swim32="+swimImage32f.depth());
//        System.out.println("swim8="+swimImage8u.depth());
//        System.out.println("renderImage="+renderImage.depth());
//        System.out.println("cameraImageBitmap="+cameraImageBitmap.depth());
//        convertScale(swimImage8u,renderImage);
//        IplImage gray = IplImage.create(swimImage8u.cvSize(), IPL_DEPTH_8U, 1);
////
//        IplImage edges = IplImage.create(gray.cvSize(), gray.depth(), gray.nChannels());
//        IplImage temp = IplImage.create(swimImage8u.cvSize(), swimImage8u.depth(), swimImage8u.nChannels());

//        cvCvtColor(swimImage8u, gray, CV_BGR2GRAY);
//        cvSmooth(gray, gray, CV_MEDIAN, MEDIAN_BLUR_FILTER_SIZE, 0, 0, 0);
//        cvLaplace(gray, edges, LAPLACIAN_FILTER_SIZE);
//        cvThreshold(edges, edges, 80, 255, CV_THRESH_BINARY_INV);
//        for (int i = 0; i < repetitions; i++) {
//            System.out.println(i);
//            cvSmooth(swimImage8u, temp, CV_BILATERAL, ksize, 0, sigmaColor, sigmaSpace);
//            cvSmooth(temp, swimImage8u, CV_BILATERAL, ksize, 0, sigmaColor, sigmaSpace);
//        }
//        temp = IplImage.create(swimImage8u.cvSize(), swimImage8u.depth(), swimImage8u.nChannels());
//        cvZero(temp);
////
////
//        cvCopy(swimImage8u, temp, edges);
////        IplImage temp1 = IplImage.create(swimImage32f.cvSize(), IPL_DEPTH_32F, swimImage32f.nChannels());
//        convertScale(temp, renderImage);
//
//        renderImage = render(temp1, glf);
//        renderImage=temp1;

//        GrayU8 gray = new GrayU8(width, height);
//        GrayU8 edges = new GrayU8(width, height);
//        Planar<GrayU8> temp = new Planar<GrayU8>(GrayU8.class,width,height,3);
//
//        ColorRgb.rgbToGray_Weighted(swimImageB,gray);
//
        sf.setTime(t1 += .05f);
        sf1.setTime(t2 += .05f);
        postInvalidate();
    }

    protected void decodeYUV420SP(int[] rgb, byte[] yuv420sp, int width, int height) {
        int frameSize = width * height;
        for (int j = 0, yp = 0; j < height; j++) {
            int uvp = frameSize + (j >> 1) * width, u = 0, v = 0;
            for (int i = 0; i < width; i++, yp++) {
                int y = (0xff & ((int) yuv420sp[yp])) - 16;
                if (y < 0)
                    y = 0;
                if ((i & 1) == 0) {
                    v = (0xff & yuv420sp[uvp++]) - 128;
                    u = (0xff & yuv420sp[uvp++]) - 128;
                }
                int y1192 = 1192 * y;

                int r = (y1192 + 1634 * v);
                int g = (y1192 - 833 * v - 400 * u);
                int b = (y1192 + 2066 * u);

                if (r < 0)
                    r = 0;
                else if (r > 262143)
                    r = 262143;
                if (g < 0)
                    g = 0;
                else if (g > 262143)
                    g = 262143;
                if (b < 0)
                    b = 0;
                else if (b > 262143)
                    b = 262143;

                rgb[yp] = 0xff000000 | ((b << 6) & 0xff0000) | ((g >> 2) & 0xff00) | ((r >> 10) & 0xff);
            }
        }
    }

//    public static void convertScale(IplImage src, IplImage dst) {
//        double scale = 0.0;
//        if (src.depth() == IPL_DEPTH_32F && dst.depth() == IPL_DEPTH_8U) {
//            scale = 255;
//        }
//        if (src.depth() == IPL_DEPTH_8U && dst.depth() == IPL_DEPTH_32F) {
//            scale = 1f / 255f;
//        }
////        System.out.println(scale);
//        cvConvertScale(src, dst, scale, 0);
//    }

    public static Bitmap render(Bitmap image, BaseFilter rf) {

        int[] bl = rf.filter(AndroidUtils.bitmapToIntArray(image), image.getWidth(), image.getHeight());
        Bitmap bm = Bitmap.createBitmap(
                bl, 0, image.getWidth(), image.getWidth(), image.getHeight(), Bitmap.Config.ARGB_8888);
        return bm;
    }

//    public static IplImage copy(IplImage image) {
//        IplImage copy = null;
//        if (image.roi() != null)
//            copy = IplImage.create(image.roi().width(), image.roi().height(), image.depth(), image.nChannels());
//        else
//            copy = IplImage.create(image.cvSize(), image.depth(), image.nChannels());
//        cvCopy(image, copy);
//        return copy;
//    }


//    public static Bitmap iplImageToBitmap(IplImage image) {
//        AndroidFrameConverter afc=new AndroidFrameConverter();
//        OpenCVFrameConverter.ToIplImage ofc = new OpenCVFrameConverter.ToIplImage();
//        return afc.convert(ofc.convert(image));
//    }
//
//    public static IplImage bitmapToIplImage(Bitmap bm) {
//        AndroidFrameConverter afc=new AndroidFrameConverter();
//        OpenCVFrameConverter.ToIplImage ofc = new OpenCVFrameConverter.ToIplImage();
//        return ofc.convert(afc.convert(bm));
//    }

    @Override
    protected void onDraw(Canvas canvas) {
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setTextSize(20);


        if (renderImageBitmap != null) {
//            bitmap.copyPixelsFromBuffer(renderImage.getByteBuffer());
            canvas.drawBitmap(renderImageBitmap, new Rect(0, 0, renderImageBitmap.getWidth(), renderImageBitmap.getHeight()), new RectF(-30, -30, FullscreenActivity.width + 60, FullscreenActivity.height + 60), null);
//        } else if (renderImageI != null) {
//            canvas.drawBitmap(iplImageToBitmap(renderImageI), new Rect(0, 0, renderImageI.width(), renderImageI.height()), new RectF(-30, -30, FullscreenActivity.width + 60, FullscreenActivity.height + 60), null);
        } else {
            System.out.println("renderimage is null!");
        }
//        String s = "FacePreview - This side up.";
//        float textWidth = paint.measureText(s);
//        canvas.drawText(s, (getWidth() - textWidth) / 2, 20, paint);
    }

}

// ----------------------------------------------------------------------

class Preview extends SurfaceView implements SurfaceHolder.Callback {
    SurfaceHolder mHolder;
    Camera mCamera;
    Camera.PreviewCallback previewCallback;

    Preview(Context context, Camera.PreviewCallback previewCallback) {
        super(context);
        this.previewCallback = previewCallback;

        // Install a SurfaceHolder.Callback so we get notified when the
        // underlying surface is created and destroyed.
        mHolder = getHolder();
        mHolder.addCallback(this);
        mHolder.setType(SurfaceHolder.SURFACE_TYPE_PUSH_BUFFERS);
    }

    public void surfaceCreated(SurfaceHolder holder) {
        // The Surface has been created, acquire the camera and tell it where
        // to draw.
        mCamera = Camera.open(1);
        try {
            mCamera.setPreviewDisplay(holder);
        } catch (IOException exception) {
            mCamera.release();
            mCamera = null;
            mCamera = Camera.open(0);
            try {
                mCamera.setPreviewDisplay(holder);
            } catch (IOException exception1) {
                mCamera.release();
                mCamera = null;
            }
        }
    }

    public void surfaceDestroyed(SurfaceHolder holder) {
        // Surface will be destroyed when we return, so stop the preview.
        // Because the CameraDevice object is not a shared resource, it's very
        // important to release it when the activity is paused.
        mCamera.stopPreview();
        mCamera.release();
        mCamera = null;
    }

    private Size getOptimalPreviewSize(List<Size> sizes, int w, int h) {
        final double ASPECT_TOLERANCE = 0.05;
        double targetRatio = (double) w / h;
        if (sizes == null)
            return null;

        Size optimalSize = null;
        double minDiff = Double.MAX_VALUE;

        int targetHeight = h;

        // Try to find an size match aspect ratio and size
        for (Size size : sizes) {
            double ratio = (double) size.width / size.height;
            if (Math.abs(ratio - targetRatio) > ASPECT_TOLERANCE)
                continue;
            if (Math.abs(size.height - targetHeight) < minDiff) {
                optimalSize = size;
                minDiff = Math.abs(size.height - targetHeight);
            }
        }

        // Cannot find the one match the aspect ratio, ignore the requirement
        if (optimalSize == null) {
            minDiff = Double.MAX_VALUE;
            for (Size size : sizes) {
                if (Math.abs(size.height - targetHeight) < minDiff) {
                    optimalSize = size;
                    minDiff = Math.abs(size.height - targetHeight);
                }
            }
        }

        //get little one
        int area1 = Integer.MAX_VALUE;
        int area2 = Integer.MAX_VALUE;
        Size secondSize = optimalSize;
        for (Size size : sizes) {
            if (size.width * size.height < area1) {
                area2 = area1;
                secondSize = optimalSize;
                area1 = size.width * size.height;
                optimalSize = size;
            }
        }
//        System.out.println(optimalSize);
//        return secondSize;
        return optimalSize;
    }

    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        // Now that the size is known, set up the camera parameters and begin
        // the preview.
        Camera.Parameters parameters = mCamera.getParameters();

        List<Size> sizes = parameters.getSupportedPreviewSizes();
        Size optimalSize = getOptimalPreviewSize(sizes, w, h);
        parameters.setPreviewSize(optimalSize.width, optimalSize.height);

        mCamera.setParameters(parameters);
        if (previewCallback != null) {
            mCamera.setPreviewCallbackWithBuffer(previewCallback);
            Camera.Size size = parameters.getPreviewSize();
            byte[] data = new byte[size.width * size.height * ImageFormat.getBitsPerPixel(parameters.getPreviewFormat()) / 8];
            mCamera.addCallbackBuffer(data);
        }
        mCamera.startPreview();
    }

}