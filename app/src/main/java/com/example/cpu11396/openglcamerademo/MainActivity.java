package com.example.cpu11396.openglcamerademo;

import android.Manifest;
import android.content.Context;
import android.content.Intent;
import android.content.pm.PackageManager;
import android.hardware.camera2.CameraManager;
import android.net.Uri;
import android.support.annotation.NonNull;
import android.support.v4.app.ActivityCompat;
import android.support.v7.app.AppCompatActivity;
import android.os.Bundle;
import android.util.DisplayMetrics;
import android.util.Log;
import android.view.View;
import android.view.Window;
import android.view.WindowManager;
import android.widget.FrameLayout;
import android.widget.ImageButton;
import android.widget.ImageView;

import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.core.Size;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileOutputStream;
import java.io.IOException;
import java.io.InputStream;

public class MainActivity extends AppCompatActivity implements PhotoGLSurfaceCallBack {

    private static final String    TAG                 = "OCV: MainActivity";
    private static final String[] PERMISSIONS = {Manifest.permission.CAMERA, Manifest.permission.WRITE_EXTERNAL_STORAGE};
    private static final int REQUEST_ALL_PERMISSION = 0;

    public static final String CAMERA_FRONT = "1";
    public static final String CAMERA_BACK = "0";

    CameraManager mCameraManager;
    private int screenWidth;
    private int screenHeight;

    private ImageView button;
    private ImageButton switchCam;

    public static String currentCamId;

    /**
     * Fields used for object detect
     */

    private static final Scalar FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    private static final Scalar    EYE_RECT_COLOR     = new Scalar(0, 0, 255, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private File mCascadeFile;
    public static CascadeClassifier mJavaDetector;
    public static DetectionBasedTracker  mNativeDetector;
    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;

    private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
        @Override
        public void onManagerConnected(int status) {
            switch (status) {
                case LoaderCallbackInterface.SUCCESS:
                {
                    // Load native library after(!) OpenCV initialization
                    System.loadLibrary("detection_based_tracker");

                    try {
                        File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);

                        // load cascade file from application resources
                        InputStream is = getResources().openRawResource(R.raw.haarcascade_frontalface_alt);
//                        InputStream is2 = getResources().openRawResource(R.raw.haarcascade_eye_tree_eyeglasses);

                        mCascadeFile = new File(cascadeDir, "haarcascade_frontalface_alt.xml");
//                        mCascadeFile2 = new File(cascadeDir, "haarcascade_eye_tree_eyeglasses.xml");

                        FileOutputStream os = new FileOutputStream(mCascadeFile);
//                        FileOutputStream os2 = new FileOutputStream(mCascadeFile2);

                        byte[] buffer = new byte[4096];
                        int bytesRead;
                        while ((bytesRead = is.read(buffer)) != -1) {
                            os.write(buffer, 0, bytesRead);
                        }
                        is.close();
                        os.close();
//
//                        byte[] buffer2 = new byte[4096];
//                        int bytesRead2;
//                        while ((bytesRead2 = is2.read(buffer2)) != -1) {
//                            os2.write(buffer2, 0, bytesRead2);
//                        }
//                        is2.close();
//                        os2.close();

                        mJavaDetector = new CascadeClassifier(mCascadeFile.getAbsolutePath());
                        if (mJavaDetector.empty()) {
                            Log.e(TAG, "Failed to load cascade classifier");
                            mJavaDetector = null;
                        } else
                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile.getAbsolutePath());

                        mNativeDetector = new DetectionBasedTracker(mCascadeFile.getAbsolutePath(), 0);

//                        mJavaDetector2 = new CascadeClassifier(mCascadeFile2.getAbsolutePath());
//                        if (mJavaDetector2.empty()) {
//                            Log.e(TAG, "Failed to load cascade classifier");
//                            mJavaDetector2 = null;
//                        } else
//                            Log.i(TAG, "Loaded cascade classifier from " + mCascadeFile2.getAbsolutePath());

//                        mNativeDetector2 = new DetectionBasedTracker(mCascadeFile2.getAbsolutePath(), 0);

                        cascadeDir.delete();

                    } catch (IOException e) {
                        e.printStackTrace();
                        Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
                    }

                } break;
                default:
                {
                    super.onManagerConnected(status);
                } break;
            }
        }
    };


    public static Intent newInstance(Context context) {
        Intent intent = new Intent(context, MainActivity.class);

        return intent;
    }

    private boolean hasPermissions(Context context, String... PERMISSIONS) {
        if (context != null && PERMISSIONS != null) {
            for (String permission : PERMISSIONS) {
                if (ActivityCompat.checkSelfPermission(context, permission) != PackageManager.PERMISSION_GRANTED) {
                    return false;
                }
            }
        }
        return true;
    }

    private PhotoGLSurfaceView glSurfaceView;

    @Override
    protected void onCreate(Bundle savedInstanceState) {

        if(!OpenCVLoader.initDebug())
        {
            Log.e(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), not working.");
        } else
        {
            Log.d(this.getClass().getSimpleName(), "  OpenCVLoader.initDebug(), working.");
        }

        super.onCreate(savedInstanceState);

        currentCamId = CAMERA_BACK;

        requestWindowFeature(Window.FEATURE_NO_TITLE);
        int ui = getWindow().getDecorView().getSystemUiVisibility();
        ui = ui | View.SYSTEM_UI_FLAG_HIDE_NAVIGATION | View.SYSTEM_UI_FLAG_FULLSCREEN | View.SYSTEM_UI_FLAG_IMMERSIVE_STICKY;
        getWindow().getDecorView().setSystemUiVisibility(ui);
        getWindow().setFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON, WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
        int screenOrientation = getResources().getConfiguration().orientation;
        getScreenDimensions();

        setContentView(R.layout.activity_main);
        glSurfaceView = findViewById(R.id.glSurfaceView);
        glSurfaceView.setLayoutParams(new FrameLayout.LayoutParams(1080, 1440));
        glSurfaceView.init(this, screenWidth, screenHeight, screenOrientation);

        button = findViewById(R.id.buttonCapture);
        button.setOnClickListener((button) -> {glSurfaceView.capturePicture();
            Mat mRgb = glSurfaceView.mRgb();
            Mat mGray = glSurfaceView.mGray();
        });

        switchCam = findViewById(R.id.switch_cam);
        switchCam.setOnClickListener(view -> switchCam());
        this.mCameraManager = (CameraManager) this.getSystemService(Context.CAMERA_SERVICE);
        if (!hasPermissions(this, PERMISSIONS)) {
            ActivityCompat.requestPermissions(this, PERMISSIONS, REQUEST_ALL_PERMISSION);
        }
    }

    private void switchCam() {
        if(currentCamId.equals("0")) {
            currentCamId = "1";
        } else {
            currentCamId = "0";
        }
        glSurfaceView.switchCamera(currentCamId);
        /*if (currentCamId.equals(CAMERA_FRONT)) {
            currentCamId = CAMERA_BACK;
            glSurfaceView.closeCamera(currentCamId);
            glSurfaceView.openCamera();

        } else {
            currentCamId = CAMERA_FRONT;
            glSurfaceView.closeCamera(currentCamId);
            glSurfaceView.openCamera();
        }*/
    }

    @Override
    public void finishTakePhotoActivity(Uri imageUri) {
        sendBroadcast(new Intent(Intent.ACTION_MEDIA_SCANNER_SCAN_FILE, imageUri));

        Intent data = new Intent();
        data.setData(imageUri);
        setResult(RESULT_OK, data);
        finish();
    }

    @Override
    protected void onResume() {
        super.onResume();
        glSurfaceView.onResume();
        if (!OpenCVLoader.initDebug()) {
            Log.d(TAG, "Internal OpenCV library not found. Using OpenCV Manager for initialization");
            OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_3_0_0, this, mLoaderCallback);
        } else {
            Log.d(TAG, "OpenCV library found inside package. Using it!");
            mLoaderCallback.onManagerConnected(LoaderCallbackInterface.SUCCESS);
        }
    }

    @Override
    protected void onPause() {
        glSurfaceView.onPause();
        super.onPause();
    }

    @Override
    public void onRequestPermissionsResult(int requestCode, @NonNull String[] permissions, @NonNull int[] grantResults) {
        switch (requestCode) {
            case REQUEST_ALL_PERMISSION:
                if (grantResults.length > 0 && permissions.length == grantResults.length) {
                    for (int i = 0; i < permissions.length; ++i) {
                        if (grantResults[i] != PackageManager.PERMISSION_GRANTED) {
                            finish();
                        }
                    }
                }
        }
    }

    private void getScreenDimensions() {
        DisplayMetrics displayMetrics = new DisplayMetrics();
        getWindowManager().getDefaultDisplay().getMetrics(displayMetrics);
        screenWidth = displayMetrics.widthPixels;
        //screenHeight = displayMetrics.heightPixels;
        screenHeight = screenWidth*4/3;
        Log.i("ScreenDimensions", "MainActivity: " + screenWidth + ", " + screenHeight);
    }
}
