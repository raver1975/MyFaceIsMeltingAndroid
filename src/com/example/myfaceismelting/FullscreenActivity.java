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

package com.example.myfaceismelting;

import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_objdetect.*;
import static com.googlecode.javacv.cpp.opencv_core.*;
import static com.googlecode.javacv.cpp.opencv_imgproc.*;
import static com.googlecode.javacv.cpp.opencv_objdetect.*;
import static com.googlecode.javacv.cpp.opencv_highgui.*;

import java.io.File;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.util.List;

import android.app.Activity;
import android.app.AlertDialog;
import android.content.Context;
import android.graphics.*;
import android.hardware.Camera;
import android.hardware.Camera.Size;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.view.SurfaceHolder;
import android.view.SurfaceView;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;

import com.googlecode.javacpp.Loader;
import com.googlecode.javacv.cpp.opencv_core.*;
import com.googlecode.javacv.cpp.opencv_core.IplImage;
import com.googlecode.javacv.cpp.opencv_objdetect;
import com.googlecode.javacv.cpp.opencv_objdetect.*;
import com.jabistudio.androidjhlabs.filter.*;
import com.jabistudio.androidjhlabs.filter.util.*;

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

    private IplImage colorImage;
    private Bitmap bitmap;
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

    public FaceView(FullscreenActivity context) throws IOException {
        super(context);

        // Preload the opencv_objdetect module to work around a known bug.
        Loader.load(opencv_objdetect.class);
        sf.setAmount(20f);
        sf.setTurbulence(1f);
        sf.setEdgeAction(sf.CLAMP);
        sf1.setEdgeAction(sf1.CLAMP);
        sf1.setAmount(30f);
        sf1.setTurbulence(1f);
        sf1.setScale(300);
        sf1.setStretch(50);
        glf.setNumLevels(100);
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

    protected void processImage(byte[] data, int width, int height) {
        // First, downsample our image and convert it into a grayscale IplImage

        if (colorImage == null || colorImage.width() != width || colorImage.height() != height) {
            colorImage = IplImage.create(width, height, IPL_DEPTH_32F, 3);
//			colorImage1 = IplImage.create(width, height, IPL_DEPTH_32F, 4);
            bitmap = Bitmap.createBitmap(colorImage.width(), colorImage.height(), Bitmap.Config.ARGB_8888);
            _temp = new int[(width * height)];
        }

        decodeYUV420SP(_temp, data, width, height);
        colorImage.getIntBuffer().put(_temp);
//        colorImage=render(colorImage,iv);


        colorImage = render(render(colorImage, sf), sf1);
//        IplImage gray = IplImage.create(colorImage.cvSize(), IPL_DEPTH_8U, 1);
//        cvCvtColor(colorImage, gray, CV_BGR2GRAY);
//        IplImage edges = IplImage.create(gray.cvSize(), gray.depth(), gray.nChannels());
//        IplImage temp = IplImage.create(colorImage.cvSize(), colorImage.depth(), colorImage.nChannels());
////
//        cvSmooth(gray, gray, CV_MEDIAN, MEDIAN_BLUR_FILTER_SIZE, 0, 0, 0);
//        cvLaplace(gray, edges, LAPLACIAN_FILTER_SIZE);
//        cvThreshold(edges, edges, 80, 255, CV_THRESH_BINARY_INV);
//        for (int i = 0; i < repetitions; i++) {
//            cvSmooth(colorImage, temp, CV_BILATERAL, ksize, 0, sigmaColor, sigmaSpace);
//            cvSmooth(temp, colorImage, CV_BILATERAL, ksize, 0, sigmaColor, sigmaSpace);
//        }
//        temp = IplImage.create(colorImage.cvSize(), colorImage.depth(), colorImage.nChannels());
//        cvZero(temp);
////
////
//        cvCopy(colorImage, temp, edges);
//        sf.setTime(t1 += .02f);
//        sf1.setTime(t2 += .02f);
//        colorImage = render(temp, glf);

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

    public static IplImage render(IplImage image, com.jabistudio.androidjhlabs.filter.util.BaseFilter rf) {
        Bitmap bm = IplImageToBitmap(image);
        int[] bl = rf.filter(AndroidUtils.bitmapToIntArray(bm), bm.getWidth(), bm.getHeight());
        bm = Bitmap.createBitmap(
                bl, 0, bm.getWidth(), bm.getWidth(), bm.getHeight(), Bitmap.Config.ARGB_8888);
        IplImage im = bitmapToIplImage(bm);
        return im;
    }

    public static IplImage copy(IplImage image) {
        IplImage copy = null;
        if (image.roi() != null)
            copy = IplImage.create(image.roi().width(), image.roi().height(), image.depth(), image.nChannels());
        else
            copy = IplImage.create(image.cvSize(), image.depth(), image.nChannels());
        cvCopy(image, copy);
        return copy;
    }

    public static Bitmap IplImageToBitmap(IplImage image) {
        Bitmap bitmap = Bitmap.createBitmap(image.width(), image.height(), Bitmap.Config.ARGB_8888);
        bitmap.copyPixelsFromBuffer(image.getByteBuffer());
        return bitmap;
    }

    public static IplImage bitmapToIplImage(Bitmap bm) {
        IplImage image = IplImage.create(bm.getWidth(), bm.getHeight(), IPL_DEPTH_32F, 3);
        bm.copyPixelsToBuffer(image.getByteBuffer());
        return image;
    }

    @Override
    protected void onDraw(Canvas canvas) {
        Paint paint = new Paint();
        paint.setColor(Color.RED);
        paint.setTextSize(20);

        String s = "FacePreview - This side up.";
        float textWidth = paint.measureText(s);
        canvas.drawText(s, (getWidth() - textWidth) / 2, 20, paint);

        if (colorImage != null && !colorImage.isNull()) {
            bitmap.copyPixelsFromBuffer(colorImage.getByteBuffer());
            canvas.drawBitmap(bitmap, null, new RectF(0, 0, FullscreenActivity.width, FullscreenActivity.height), null);
        } else {
            System.out.println("colorimage is null!");
        }
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
        mCamera = Camera.open();
        try {
            mCamera.setPreviewDisplay(holder);
        } catch (IOException exception) {
            mCamera.release();
            mCamera = null;
            // TODO: add more exception handling logic here
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
        System.out.println(optimalSize);
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