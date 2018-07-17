package com.example.cpu11396.openglcamerademo;

import android.content.Context;
import android.content.res.Configuration;
import android.net.Uri;
import android.opengl.GLSurfaceView;
import android.util.AttributeSet;
import android.util.Log;
import android.view.SurfaceHolder;

import org.opencv.core.Mat;

import javax.microedition.khronos.egl.EGL10;
import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.egl.EGLContext;
import javax.microedition.khronos.egl.EGLDisplay;
import javax.microedition.khronos.egl.EGLSurface;

public class PhotoGLSurfaceView extends GLSurfaceView {
    private PhotoRenderer photoRenderer;
    private PhotoGLSurfaceCallBack callBack;
    private EGLContext eglContext;
    private boolean setUpFinish;
    private int screenWidth;
    private int screenHeight;
    private int screenOrientation;

    public PhotoGLSurfaceView(Context context) {
        super(context);
    }

    public PhotoGLSurfaceView(Context context, AttributeSet attrs) {
        super(context, attrs);
    }

    public void init(PhotoGLSurfaceCallBack callBack, int screenWidth, int screenHeight, int screenOrientation) {
        this.screenHeight = screenHeight;
        this.screenWidth = screenWidth;
        this.screenOrientation = screenOrientation;
        setUpFinish = false;
        this.callBack = callBack;
        photoRenderer = new PhotoRenderer(this,screenWidth, screenHeight, screenOrientation, "0");
        setEGLContextClientVersion(2);
        setRenderer(photoRenderer);
        setRenderMode(GLSurfaceView.RENDERMODE_WHEN_DIRTY);
        Log.i("ScreenDimensions", "PhotoGLSurfaceView: " + screenWidth + ", " + screenHeight);

    }

    @Override
    public void surfaceCreated(SurfaceHolder holder) {
        super.surfaceCreated(holder);
    }

    @Override
    public void surfaceDestroyed(SurfaceHolder holder) {
        super.surfaceDestroyed(holder);
    }

    @Override
    public void surfaceChanged(SurfaceHolder holder, int format, int w, int h) {
        super.surfaceChanged(holder, format, w, h);
    }



    @Override
    public void onResume() {
        super.onResume();
        photoRenderer.onResume();
    }

    @Override
    public void onPause() {
        photoRenderer.onPause();
        super.onPause();
    }

    public void finishTakePhotoActivity(Uri imageURI) {
        callBack.finishTakePhotoActivity(imageURI);
    }

    /**
     * User can only capture picture when the camera is opened
     *
     * @param b: if true: we can capture photo */
    public void setSetUpFinish(boolean b) {
        setUpFinish = b;
    }


    public void capturePicture() {
        if (setUpFinish)
            photoRenderer.capturePicture();
    }

    public void closeCamera(String newCamId) {
        photoRenderer.closeCamera(newCamId);
    }

    public void openCamera() {
        //photoRenderer = new PhotoRenderer(this,screenWidth, screenHeight, screenOrientation, newCamId);
        photoRenderer.openCamera();
    }

    public Mat mRgb() {
        return photoRenderer.mRgb;
    }

    public Mat mGray() {
        return photoRenderer.mGray;
    }

    public void setmRgb(Mat mRgb) {
        photoRenderer.mRgb = mRgb;
    }

    private class SampleContextFactory implements EGLContextFactory {
        private int EGL_CONTEXT_CLIENT_VERSION = 0x3098;

        public EGLContext createContext(EGL10 egl, EGLDisplay display, EGLConfig config) {
            int[] attrib_list = {EGL_CONTEXT_CLIENT_VERSION, 3,
                    EGL10.EGL_NONE};

            eglContext = egl.eglCreateContext(display, config, EGL10.EGL_NO_CONTEXT,
                    3 != 0 ? attrib_list : null);
            return eglContext;
        }

        public void destroyContext(EGL10 egl, EGLDisplay display,
                                   EGLContext context) {
            if (!egl.eglDestroyContext(display, context)) {
                Log.e("DefaultContextFactory", "display:" + display + " context: " + context);
            }
        }
    }
}
