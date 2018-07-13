package com.example.cpu11396.openglcamerademo;

import android.annotation.SuppressLint;
import android.content.Context;
import android.content.res.Configuration;
import android.graphics.Bitmap;
import android.graphics.Matrix;
import android.graphics.Point;
import android.graphics.SurfaceTexture;
import android.hardware.camera2.CameraAccessException;
import android.hardware.camera2.CameraCaptureSession;
import android.hardware.camera2.CameraCharacteristics;
import android.hardware.camera2.CameraDevice;
import android.hardware.camera2.CameraManager;
import android.hardware.camera2.CaptureRequest;
import android.hardware.camera2.params.StreamConfigurationMap;
import android.net.Uri;
import android.opengl.GLES11Ext;
import android.opengl.GLES30;
import android.opengl.GLSurfaceView;
import android.os.Environment;
import android.os.Handler;
import android.os.HandlerThread;
import android.support.annotation.NonNull;
import android.util.Log;
import android.util.Size;
import android.view.Surface;

import org.opencv.android.Utils;
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.MatOfRect;
import org.opencv.core.Rect;
import org.opencv.core.Scalar;
import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileOutputStream;
import java.io.IOException;
import java.nio.ByteBuffer;
import java.nio.ByteOrder;
import java.nio.FloatBuffer;
import java.text.SimpleDateFormat;
import java.util.Collections;
import java.util.Date;
import java.util.Locale;
import java.util.concurrent.Semaphore;
import java.util.concurrent.TimeUnit;

import javax.microedition.khronos.egl.EGLConfig;
import javax.microedition.khronos.opengles.GL10;

import static org.opencv.core.CvType.CV_8UC4;

public class PhotoRenderer implements GLSurfaceView.Renderer, SurfaceTexture.OnFrameAvailableListener
        , CallBackListener {
    private PhotoGLSurfaceView photoGLSurfaceView;
    private final String VERTEX_SHADER = "" +
            "precision mediump float;" +
            "attribute vec4 aPosition;" +
            "attribute vec4 aTextureCoordinate;" +
            "varying vec2 texCoord;" +
            "void main() {" +
            "texCoord = aTextureCoordinate.xy;" +
            "gl_Position = aPosition;" +
            "}";
    private final String FRAGMENT_SHADER = "" +
            "#extension GL_OES_EGL_image_external : require\n" +
            "precision mediump float;" +
            "uniform samplerExternalOES sTexture;" +
            "uniform mediump mat4 colorMatrix;" +
            "uniform mediump float intensity;" +
            "varying vec2 texCoord;" +
            "void main() {" +
            "gl_FragColor = texture2D(sTexture, texCoord);" +
           // "vec4 outputColor = color * colorMatrix;" +
            //"gl_FragColor = (intensity * outputColor) + ((1.0 - intensity) * color);" +
            "}";
    private final float[] COLOR_MATRIX = {
            1, 1, 1, 0f,
            1, 1, 1, 0f,
            1, 1, 1, 0f,
            0, 0, 0, 1f
    };

    private final float[] TEXTURE_COORDINATE = {
            1.0f, 1.0f,
            0.0f, 1.0f,
            1.0f, 0.0f,
            0.0f, 0.0f
    };

    private final float[] VERTEX_POSITION_PORTRAIT_FRONT = {
            /**
             * Front camera
            */
            -1.0f, 1.0f,
            -1.0f, -1.0f,
            1.0f, 1.0f,
            1.0f, -1.0f,
            /**
             * Default mapping
            */
//            1.0f, 1.0f,
//            -1.0f, 1.0f,
//            1.0f, -1.0f,
//            -1.0f, -1.0f,
    };

    private final float[] VERTEX_POSITION_PORTRAIT_BACK = {
            /**
             * Back camera
            */
            -1.0f, -1.0f,
            -1.0f, 1.0f,
            1.0f, -1.0f,
            1.0f, 1.0f,
    };
    private final float[] VERTEX_POSITION_LANDSCAPE_FRONT = {
            /**
             * Front camera
            */
            -1.0f, -1.0f,
            1.0f, -1.0f,
            -1.0f, 1.0f,
            1.0f, 1.0f,
    };
    private final float[] VERTEX_POSITION_LANDSCAPE_BACK = {
            /**
             * Back camera
            */
            1.0f, -1.0f,
            -1.0f, -1.0f,
            1.0f, 1.0f,
            -1.0f, 1.0f,
    };
    private int[] hTex;
    private FloatBuffer pVertex;
    private FloatBuffer pTextureCoordinate;
    private int program;

    private SurfaceTexture surfaceTexture;
    private Surface textureSurface;

    private boolean GLInit = false;
    private boolean updateSurfaceTexture = false;

    private CameraDevice cameraDevice;
    private CameraCaptureSession captureSession;
    private CaptureRequest.Builder previewRequestBuilder;
    private String cameraID;

    private Size previewSize;

    private HandlerThread backgroundThread;
    private Handler backgroundHandler;
    private Semaphore cameraOpenCloseLock = new Semaphore(1);

    private final int SCREEN_WIDTH;
    private final int SCREEN_HEIGHT;
    private final int SCREEN_ORIENTATION;

    /**
     * Fields use for object detect
     */

    private static final Scalar FACE_RECT_COLOR     = new Scalar(0, 255, 0, 255);
    private static final Scalar    EYE_RECT_COLOR     = new Scalar(0, 0, 255, 255);
    public static final int        JAVA_DETECTOR       = 0;
    public static final int        NATIVE_DETECTOR     = 1;

    private File mCascadeFile;
    private CascadeClassifier mJavaDetector;
    private DetectionBasedTracker  mNativeDetector;
    private int                    mDetectorType       = JAVA_DETECTOR;
    private String[]               mDetectorName;
    public Mat mRgb;
    public Mat mGray;

    public PhotoRenderer(PhotoGLSurfaceView photoGLSurfaceView, int screenWidth, int screenHeight, int screenOrientation, String cameraID) {
        SCREEN_WIDTH = screenWidth;
        SCREEN_HEIGHT = screenHeight;
        SCREEN_ORIENTATION = screenOrientation;
        this.photoGLSurfaceView = photoGLSurfaceView;
        this.cameraID = cameraID;

        previewSize = new Size(1080,1440);

        pVertex = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        if (screenOrientation == Configuration.ORIENTATION_PORTRAIT && cameraID.equals("1"))
            pVertex.put(VERTEX_POSITION_PORTRAIT_FRONT);
        else if(screenOrientation == Configuration.ORIENTATION_PORTRAIT && cameraID.equals("0"))
            pVertex.put(VERTEX_POSITION_PORTRAIT_BACK);
        else if (screenOrientation == Configuration.ORIENTATION_LANDSCAPE && cameraID.equals("1"))
            pVertex.put(VERTEX_POSITION_LANDSCAPE_FRONT);
        else
            pVertex.put(VERTEX_POSITION_LANDSCAPE_BACK);
        pVertex.position(0);

        pTextureCoordinate = ByteBuffer.allocateDirect(8 * 4).order(ByteOrder.nativeOrder()).asFloatBuffer();
        pTextureCoordinate.put(TEXTURE_COORDINATE);
        pTextureCoordinate.position(0);
    }

    public void onResume() {
        startBackgroundThread();
    }

    public void onPause() {
        GLInit = false;
        updateSurfaceTexture = false;
        closeCamera();
        stopBackgroundThread();
    }

    private void stopBackgroundThread() {
        backgroundThread.quitSafely();
        try {
            backgroundThread.join();
            backgroundThread = null;
            backgroundHandler = null;
        } catch (InterruptedException e) {
            e.printStackTrace();
            throw new RuntimeException("Interrupted exception " + e.getMessage());
        }
    }

    private void startBackgroundThread() {
        backgroundThread = new HandlerThread("CameraBackground");
        backgroundThread.start();
        backgroundHandler = new Handler(backgroundThread.getLooper());
    }

    @Override
    public void onSurfaceCreated(GL10 gl, EGLConfig config) {
        GLES30.glClearColor(1f, 1f, 1f, 1f);
        Point p = new Point();
        photoGLSurfaceView.getDisplay().getRealSize(p);
        calculatePreviewSize(p);

        initTex();
        surfaceTexture = new SurfaceTexture(hTex[0]);
        surfaceTexture.setOnFrameAvailableListener(this);

        program = loadShader(VERTEX_SHADER, FRAGMENT_SHADER);

        openCamera(cameraID);

        GLInit = true;
    }

    @Override
    public void onSurfaceChanged(GL10 gl, int width, int height) {
        GLES30.glViewport(0, 0, width, height);
    }

    @Override
    public void onDrawFrame(GL10 gl) {
        if (!GLInit)
            return;
        GLES30.glClear(GLES30.GL_COLOR_BUFFER_BIT);

        synchronized (this) {
            if (updateSurfaceTexture) {
                surfaceTexture.updateTexImage();
                updateSurfaceTexture = false;
            }
        }
        GLES30.glUseProgram(program);

        GLES30.glActiveTexture(GLES30.GL_TEXTURE0);
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, hTex[0]);

        int positionHandle = GLES30.glGetAttribLocation(program, "aPosition");
        int textureCoordHandle = GLES30.glGetAttribLocation(program, "aTextureCoordinate");

        GLES30.glVertexAttribPointer(positionHandle, 2, GLES30.GL_FLOAT, false, 4 * 2, pVertex);
        GLES30.glVertexAttribPointer(textureCoordHandle, 2, GLES30.GL_FLOAT, false, 4 * 2, pTextureCoordinate);

        GLES30.glEnableVertexAttribArray(positionHandle);
        GLES30.glEnableVertexAttribArray(textureCoordHandle);

//        int colorMatrixHandle = GLES30.glGetUniformLocation(program, "colorMatrix");
//        int intensityHandle = GLES30.glGetUniformLocation(program, "intensity");
//        GLES30.glUniform1f(intensityHandle, 1f);
//        GLES30.glUniformMatrix4fv(colorMatrixHandle, 1, false, COLOR_MATRIX, 0);

        GLES30.glUniform1i(GLES30.glGetUniformLocation(program, "sTexture"), 0);
        GLES30.glDrawArrays(GLES30.GL_TRIANGLE_STRIP, 0, 4);
        GLES30.glFlush();

        if (isCapture) {
            File rootFile;
            File mFile;
            rootFile = new File(Environment
                    .getExternalStorageDirectory()
                    .getAbsolutePath() + "/MyAppChat/Images");
            if (!rootFile.exists())
                rootFile.mkdirs();
            mFile = new File(rootFile.getPath() + "/Image_"
                    + new SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(new Date())
                    + ".jpeg");
            Bitmap bmp;
            int width = previewSize.getWidth();
            int height = previewSize.getHeight();

            ByteBuffer buffer = ByteBuffer.allocateDirect(width * height * 4);
            GLES30.glReadPixels(0, 0, width, height, GLES30.GL_RGBA, GLES30.GL_UNSIGNED_BYTE, buffer);
            buffer.order(ByteOrder.nativeOrder());

            Mat mRawRgb = new Mat(height, width, CV_8UC4, buffer);
            mRgb = new Mat();

            /**
             * If using back camera, flip Mat to correct orientation
             */
            if(cameraDevice.getId().equals("1")) {
                Core.flip(mRawRgb, mRgb, 0);
            }
            else {
                mRgb = mRawRgb;
            }

            mGray = new Mat();
            Imgproc.cvtColor(mRgb, mGray, Imgproc.COLOR_RGB2GRAY);

            mNativeDetector = MainActivity.mNativeDetector;
            mJavaDetector = MainActivity.mJavaDetector;

            mNativeDetector.setMinFaceSize(360);
            MatOfRect faces = new MatOfRect();

            if (mDetectorType == JAVA_DETECTOR) {
                if (mJavaDetector != null)
                    mJavaDetector.detectMultiScale(mGray, faces, 1.1, 4, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
                            new org.opencv.core.Size(360, 360), new org.opencv.core.Size());

//                if (mJavaDetector2 != null)
//                    mJavaDetector2.detectMultiScale(mGray, eyes, 1.1, 5, 2, // TODO: objdetect.CV_HAAR_SCALE_IMAGE
//                            new Size(mAbsoluteFaceSize/3, mAbsoluteFaceSize/3), new Size());
            }
            else if (mDetectorType == NATIVE_DETECTOR) {
                if (mNativeDetector != null)
                    mNativeDetector.detect(mGray, faces);

//                if (mNativeDetector2 != null)
//                    mNativeDetector2.detect(mGray, eyes);
            }
            else {
                Log.e("OCV: PhotoRenderer", "Detection method is not selected!");
            }

            Rect[] facesArray = faces.toArray();
            for (int i = 0; i < facesArray.length; i++)
                Imgproc.rectangle(mRgb, facesArray[i].tl(), facesArray[i].br(), FACE_RECT_COLOR, 4);


            bmp = Bitmap.createBitmap(width, height, Bitmap.Config.ARGB_8888);
//            bmp.copyPixelsFromBuffer(buffer);
//            // Flip the bitmap, as the buffer of GLES is different from the buffer order
//            // of bitmap

            Utils.matToBitmap(mRgb,bmp);

//            bmp = flipBitmap(bmp);
            FileOutputStream fileOutputStream = null;
            try {
                fileOutputStream = new FileOutputStream(mFile.getPath());
                bmp.compress(Bitmap.CompressFormat.JPEG, 90, fileOutputStream);
            } catch (FileNotFoundException e) {
                e.printStackTrace();
                throw new RuntimeException("File not found exception " + e.getMessage());
            } finally {
                if (fileOutputStream != null) {
                    try {
                        fileOutputStream.close();
                    } catch (IOException e) {
                        e.printStackTrace();
                    }
                }
            }

            isCapture = false;
            bmp.recycle();
            mRawRgb.release();
            mRgb.release();
            mGray.release();
            //saveFileSuccessfully(mFile);
        }
    }

    /**
     * Bitmap generated by GLES is flipped upside down, so we flip it again
     */
    private Bitmap flipBitmap(Bitmap bitmap) {
        Matrix matrix = new Matrix();
        matrix.postScale(1, -1, bitmap.getWidth() / 2f, bitmap.getHeight() / 2f);
        //matrix.postRotate(180);
        return Bitmap.createBitmap(bitmap, 0, 0, bitmap.getWidth(), bitmap.getHeight(), matrix, true);
    }

    @Override
    public synchronized void onFrameAvailable(SurfaceTexture surfaceTexture) {
        updateSurfaceTexture = true;
        photoGLSurfaceView.requestRender();

    }

    private void initTex() {
        hTex = new int[1];
        GLES30.glGenTextures(1, hTex, 0);
        GLES30.glBindTexture(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, hTex[0]);

        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_S, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_WRAP_T, GLES30.GL_CLAMP_TO_EDGE);
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MIN_FILTER, GLES30.GL_NEAREST);
        GLES30.glTexParameteri(GLES11Ext.GL_TEXTURE_EXTERNAL_OES, GLES30.GL_TEXTURE_MAG_FILTER, GLES30.GL_LINEAR);
    }

    private static int loadShader(String vertexShader, String fragmentShader) {
        int vShader = GLES30.glCreateShader(GLES30.GL_VERTEX_SHADER);
        GLES30.glShaderSource(vShader, vertexShader);
        GLES30.glCompileShader(vShader);
        int[] compiled = new int[1];
        // Get the value of parameter(GL_COMPILE_STATUS) for a specific
        // shader object (vShader)
        GLES30.glGetShaderiv(vShader, GLES30.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            Log.i("testing", "Cannot compile vertex shader " + GLES30.glGetShaderInfoLog(vShader));
            GLES30.glDeleteShader(vShader);
            vShader = 0;
        }

        int fShader = GLES30.glCreateShader(GLES30.GL_FRAGMENT_SHADER);
        GLES30.glShaderSource(fShader, fragmentShader);
        GLES30.glCompileShader(fShader);
        GLES30.glGetShaderiv(fShader, GLES30.GL_COMPILE_STATUS, compiled, 0);
        if (compiled[0] == 0) {
            Log.i("testing", "Cannot compile fragment shader " + GLES30.glGetShaderInfoLog(fShader));
            GLES30.glDeleteShader(fShader);
            fShader = 0;
        }

        int program = GLES30.glCreateProgram();
        GLES30.glAttachShader(program, vShader);
        GLES30.glAttachShader(program, fShader);
        GLES30.glLinkProgram(program);

        return program;
    }

    /**
     * Choose the previewSize. This previewSize is used for surface texture, bitmap operations
     *
     * @param p
     */
    private void calculatePreviewSize(Point p) {
        CameraManager cameraManager = (CameraManager) photoGLSurfaceView.getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            assert cameraManager != null;
            for (String id : cameraManager.getCameraIdList()) {
                CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(id);
                if (cameraCharacteristics.get(CameraCharacteristics.LENS_FACING) == CameraCharacteristics.LENS_FACING_FRONT)
                    continue;

                cameraID = id;
                StreamConfigurationMap map = cameraCharacteristics.get(CameraCharacteristics.SCALER_STREAM_CONFIGURATION_MAP);
                assert map != null;

                Size[] sizes = map.getOutputSizes(SurfaceTexture.class);
                previewSize = chooseOptimalSize(sizes, SCREEN_WIDTH, SCREEN_HEIGHT);
                previewSize = new Size(1080,1440);
            }
        } catch (CameraAccessException e) {
            e.printStackTrace();
            throw new RuntimeException("Camera access exception " + e.getMessage());
        } catch (IllegalArgumentException e) {
            throw new RuntimeException("Illegal argument exception " + e.getMessage());
        } catch (SecurityException e) {
            throw new RuntimeException("Security exception " + e.getMessage());
        }
    }

    /**
     * For screen orientation portrait:
     * <p>
     * Screen width = 720
     * Screen height = 1280
     * --> width < height
     * </p>
     * For screen orientation landscape:
     * <p>
     * Screen width = 1280
     * Screen height = 720
     * --> width > height
     * </p>
     * Surface texture's sizes :
     * <p>
     * width = 1280
     * height = 720
     * --> width > height
     * </p>
     * We need to map these when compare surface texture's size and screen size
     * And return result will have width < height (for portrait orientation)
     */
    private Size chooseOptimalSize(Size[] pSize, int screen_width, int screen_height) {
        sortSizes(pSize);
        int width = pSize[0].getWidth();
        int height = pSize[0].getHeight();
        for (Size size : pSize) {
            Log.i("sizzzze", size.getWidth() + ", " + size.getHeight());
            if (SCREEN_ORIENTATION == Configuration.ORIENTATION_PORTRAIT) {
                // If the new surface texture's height <= screen width (mapped as mentioned above)
                // and if this new surface texture's height is > than previous height
                if (size.getHeight() <= screen_width && height < size.getHeight())
                    height = size.getHeight();
                // Same
                if (size.getWidth() <= screen_height && width < size.getWidth())
                    width = size.getWidth();

                    // If surface texture's height > screen width OR
                    // surface texture's width > screen height
                else if (size.getHeight() > screen_width || size.getWidth() > screen_height) {
                    if (size.getHeight() > screen_width)
                        height = screen_width; // maximum width(we return Size(height, width)) cannot be larger than screen size
                    if (size.getWidth() > screen_height) {
                        width = size.getWidth(); // maximum height CAN be larger than screen size
                        return new Size(height, width);
                    }
                }
            } else if (SCREEN_ORIENTATION == Configuration.ORIENTATION_LANDSCAPE) {
                if (size.getHeight() <= screen_height && height < size.getHeight())
                    height = size.getHeight();
                if (size.getWidth() <= screen_width && width < size.getWidth())
                    width = size.getWidth();
                else if (size.getHeight() > screen_height || size.getWidth() > screen_width) {
                    if (size.getWidth() > screen_width)
                        width = screen_width; // maximum width(we return Size(width, height)) cannot be larger than screen size
                    if (size.getHeight() > screen_height) {
                        height = size.getHeight();
                        return new Size(width, height);
                    }
                }
            }
        }
        if (SCREEN_ORIENTATION == Configuration.ORIENTATION_PORTRAIT)
            return new Size(height, width);
        else return new Size(width, height);
    }

    /**
     * We sort supported sizes of surface texture based on each size's area
     */
    private void sortSizes(Size[] pSize) {
        int n = pSize.length;
        for (int i = 1; i < n; ++i) {
            long key = pSize[i].getWidth() * pSize[i].getHeight();
            Size temp = pSize[i];
            int j = i - 1;

            /* Move elements of arr[0..i-1], that are
               greater than key, to one position ahead
               of their current position */
            while (j >= 0 && (pSize[j].getWidth() * pSize[j].getHeight()) > key) {
                pSize[j + 1] = pSize[j];
                j = j - 1;
            }
            pSize[j + 1] = temp;
        }
    }

    @Override
    public void saveFileSuccessfully(File mFile) {
        Uri imageURI = Uri.fromFile(mFile);
        photoGLSurfaceView.finishTakePhotoActivity(imageURI);
    }

    @SuppressLint("MissingPermission")
    public void openCamera(String newCamId) {

        CameraManager cameraManager = (CameraManager) photoGLSurfaceView.getContext().getSystemService(Context.CAMERA_SERVICE);
        try {
            assert cameraManager != null;
            CameraCharacteristics cameraCharacteristics = cameraManager.getCameraCharacteristics(cameraID);
            if (!cameraOpenCloseLock.tryAcquire(2500, TimeUnit.MILLISECONDS)) {
                throw new RuntimeException("Time out waiting to lock camera opening.");
            }
            cameraManager.openCamera(newCamId, stateCallBack, backgroundHandler);
        } catch (CameraAccessException e) {
            e.printStackTrace();
            throw new RuntimeException("Camera access exception " + e.getMessage());
        } catch (InterruptedException e) {
            e.printStackTrace();
            throw new RuntimeException("Interrupted exception " + e.getMessage());
        }
    }

    public void closeCamera() {
        try {
            cameraOpenCloseLock.acquire();
            if (captureSession != null) {
                captureSession.close();
                captureSession = null;
            }
            if (cameraDevice != null) {
                cameraDevice.close();
                cameraDevice = null;
            }
        } catch (InterruptedException e) {
            e.printStackTrace();
            throw new RuntimeException("Interrupted exception " + e.getMessage());
        } finally {
            cameraOpenCloseLock.release();
        }
    }


    private final CameraDevice.StateCallback stateCallBack = new CameraDevice.StateCallback() {
        @Override
        public void onOpened(@NonNull CameraDevice camera) {
            cameraOpenCloseLock.release();
            cameraDevice = camera;
            photoGLSurfaceView.setSetUpFinish(true);
            createCameraPreviewSession();
        }

        @Override
        public void onDisconnected(@NonNull CameraDevice camera) {
            cameraOpenCloseLock.release();
            cameraDevice.close();
            cameraDevice = null;
            photoGLSurfaceView.setSetUpFinish(false);
        }

        @Override
        public void onError(@NonNull CameraDevice camera, int error) {
            cameraOpenCloseLock.release();
            cameraDevice.close();
            cameraDevice = null;
            photoGLSurfaceView.setSetUpFinish(false);
        }
    };

    private void createCameraPreviewSession() {
        try {
            surfaceTexture.setDefaultBufferSize(previewSize.getWidth(), previewSize.getHeight());

            textureSurface = new Surface(surfaceTexture);

            previewRequestBuilder = cameraDevice.createCaptureRequest(CameraDevice.TEMPLATE_PREVIEW);
            previewRequestBuilder.addTarget(textureSurface);
            cameraDevice.createCaptureSession(Collections.singletonList(textureSurface),
                    new CameraCaptureSession.StateCallback() {
                        @Override
                        public void onConfigured(@NonNull CameraCaptureSession session) {
                            if (cameraDevice == null)
                                return;
                            captureSession = session;
                            try {
                                previewRequestBuilder.set(CaptureRequest.CONTROL_AF_MODE, CaptureRequest.CONTROL_AF_MODE_CONTINUOUS_PICTURE);
                                previewRequestBuilder.set(CaptureRequest.CONTROL_AE_MODE, CaptureRequest.CONTROL_AE_MODE_ON_AUTO_FLASH);

                                captureSession.setRepeatingRequest(previewRequestBuilder.build(), null, backgroundHandler);
                            } catch (CameraAccessException e) {
                                e.printStackTrace();
                            }
                        }

                        @Override
                        public void onConfigureFailed(@NonNull CameraCaptureSession session) {

                        }
                    }, null);
        } catch (CameraAccessException e) {
            e.printStackTrace();
            throw new RuntimeException("Camera access exception " + e.getMessage());
        }
    }

    private boolean isCapture = false;

    public void capturePicture() {
        isCapture = true;
    }

}