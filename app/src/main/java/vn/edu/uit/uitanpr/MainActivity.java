package vn.edu.uit.uitanpr;

import android.app.Activity;
import android.content.Context;
import android.graphics.Bitmap;
import android.graphics.Canvas;
import android.graphics.Color;
import android.graphics.Paint;
import android.hardware.Camera;
import android.os.Bundle;
import android.os.Environment;

import android.util.Log;
import android.view.*;
import android.widget.*;

import java.io.*;
import java.util.*;

import org.opencv.core.*;
import org.opencv.android.BaseLoaderCallback;
import org.opencv.android.LoaderCallbackInterface;
import org.opencv.android.OpenCVLoader;

import org.opencv.imgproc.Imgproc;
import org.opencv.objdetect.CascadeClassifier;

import vn.edu.uit.uitanpr.common.PlatesListAdapter;
import vn.edu.uit.uitanpr.common.Utils;
import vn.edu.uit.uitanpr.interfaces.OnTaskCompleted;
import vn.edu.uit.uitanpr.views.CameraPreview;


public class MainActivity extends Activity implements OnTaskCompleted {
	private CameraPreview cameraPreview;
	private RelativeLayout layout;
	private PlateView plateView;
	PlatesListAdapter adapter;
	Utils utils;
	public boolean isRunningTask = false;
	public boolean isFail = false;

	public static final String PACKAGE_NAME = "vn.edu.uit.uitanpr";
	public static final String DATA_PATH = Environment.getExternalStorageDirectory().toString() + "/UIT-ANPR/";

	public static final String lang = "eng";

	private static final String TAG = "MainActivity.java";

	List<Point> platePointList;
	TextView foundNumberPlate;


	private Mat mRgba;
	private Mat mGray;
	private File mCascadeFile;
	private CascadeClassifier mJavaDetector;

	MatOfRect plates;

	private float mRelativePlateSize = 0.2f;
	private int mAbsolutePlateSize = 0;


	private BaseLoaderCallback mLoaderCallback = new BaseLoaderCallback(this) {
		@Override
		public void onManagerConnected(int status) {
			switch (status) {
			case LoaderCallbackInterface.SUCCESS: {
				Log.i(TAG, "OpenCV loaded successfully");

				try {
					// Load Haar training result file from application resources
                    // This file from opencv_traincascade tool.
                    // Load res/cascade-europe.xml file
                    InputStream is = getResources().openRawResource(R.raw.europe);

					File cascadeDir = getDir("cascade", Context.MODE_PRIVATE);
					mCascadeFile = new File(cascadeDir, "europe.xml"); // Load XML file according to R.raw.cascade
					FileOutputStream os = new FileOutputStream(mCascadeFile);

					byte[] buffer = new byte[4096];
					int bytesRead;
					while ((bytesRead = is.read(buffer)) != -1) {
						os.write(buffer, 0, bytesRead);
					}
					is.close();
					os.close();

					mJavaDetector = new CascadeClassifier(
							mCascadeFile.getAbsolutePath());
					if (mJavaDetector.empty()) {
						Log.e(TAG, "Failed to load cascade classifier");
						mJavaDetector = null;
					} else
						Log.i(TAG, "Loaded cascade classifier from "
								+ mCascadeFile.getAbsolutePath());

					cascadeDir.delete();

				} catch (IOException e) {
					e.printStackTrace();
					Log.e(TAG, "Failed to load cascade. Exception thrown: " + e);
				}
			}
				break;
			case LoaderCallbackInterface.INIT_FAILED:
			{
				
			}
				break;
			default: {
				super.onManagerConnected(status);
			}
				break;
			}
		}
	};

	/** Called when the activity is first created. */
	@Override
	public void onCreate(Bundle savedInstanceState) {
		super.onCreate(savedInstanceState);
		requestWindowFeature(Window.FEATURE_NO_TITLE);
		super.onCreate(savedInstanceState);

		getWindow().addFlags(WindowManager.LayoutParams.FLAG_FULLSCREEN);
		getWindow().addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON);
		setContentView(R.layout.activity_main);
		Boolean checkOpenCV = OpenCVLoader.initAsync(
                                    OpenCVLoader.OPENCV_VERSION_2_4_9,
                                    getApplicationContext(),
                                    mLoaderCallback);
		if(checkOpenCV)
		{
			try {
				layout = (RelativeLayout) findViewById(R.id.mainFrame);
				plateView = new PlateView(this);
				cameraPreview = new CameraPreview(this, plateView);
				layout.addView(cameraPreview, 1);
				layout.addView(plateView, 2);
			} catch (Exception e1) {
                Log.e("MissingOpenCVManager",e1.toString());
			}

			utils = new Utils(getBaseContext());
			platePointList = new ArrayList<Point>();


		}


	}

	@Override
	public void onPause() {
		super.onPause();
	}

	@Override
	public void onResume() {
		super.onResume();
		OpenCVLoader.initAsync(OpenCVLoader.OPENCV_VERSION_2_4_9,
                this, mLoaderCallback);
	}

	public void onDestroy() {
  		super.onDestroy();
	}

    // This class is for Viewing a detected plate

	public class PlateView extends View implements Camera.PreviewCallback,
			OnTaskCompleted {
		public static final int SUBSAMPLING_FACTOR = 4;
		Rect[] platesArray;
		Bitmap og;
		List<Point> currentPlatePointList = new ArrayList<Point>();
		List<Rect> currentPlates = new ArrayList<Rect>();

		public PlateView(MainActivity context) throws IOException {
			super(context);
		}

		public void onPreviewFrame(final byte[] data, final Camera camera) {
						
			try {
				Camera.Size size = camera.getParameters().getPreviewSize();

                // Get image from camera and process it to see
                // are there any plate on it ?
				processImage(data, size.width, size.height);
				camera.addCallbackBuffer(data);

			} catch (RuntimeException e) {
				// The camera has probably just been released, ignore.
			}
		}

		protected void processImage(byte[] data, int width, int height) {


			// First, downsample our image and convert it into a grayscale

			int f = SUBSAMPLING_FACTOR;
			mRgba = new Mat(height, width, CvType.CV_8UC4);
			mGray = new Mat(height, width, CvType.CV_8UC1);

			Mat mYuv = new Mat(height + height / 2, width, CvType.CV_8UC1);
			mYuv.put(0, 0, data);

			Imgproc.cvtColor(mYuv, mGray, Imgproc.COLOR_YUV420sp2GRAY);
			Imgproc.cvtColor(mYuv, mRgba, Imgproc.COLOR_YUV2RGB_NV21, 3);

			if (mAbsolutePlateSize == 0) {
				int heightGray = mGray.rows();
				if (Math.round(heightGray * mRelativePlateSize) > 0) {
					mAbsolutePlateSize = Math.round(heightGray
							* mRelativePlateSize);
				}
			}

            // This variable is used to to store the detected plates in the result
			plates = new MatOfRect();

			if (mJavaDetector != null)
				mJavaDetector.detectMultiScale(
                        mGray,
                        plates,
                        1.1,
                        2,
                        2,
						new Size(mAbsolutePlateSize, mAbsolutePlateSize),
						new Size()
                );

			postInvalidate();
		}

		@Override
		protected void onDraw(Canvas canvas) {
			Paint paint = new Paint();
			paint.setColor(Color.GREEN);

			paint.setTextSize(20);
			if (plates != null) {
				paint.setStrokeWidth(5);
				paint.setStyle(Paint.Style.STROKE);

				platesArray = plates.toArray();
				boolean isHasNewPlate = false;
				currentPlates.clear();

				for (int i = 0; i < platesArray.length; i++) {
					int x = platesArray[i].x;
                    int y = platesArray[i].y;
                    int w = platesArray[i].width;
                    int h = platesArray[i].height;

                    // Draw a Green Rectangle surrounding the Number Plate !
                    // Congratulations ! You found the plate area :-)

					canvas.drawRect(x, y, (x + w), (y + h), paint);

                    Log.i("Plate found"," Found a plate !!!");
					
					// isNewPlate?
					Point platePoint = new Point(platesArray[i].x,
							platesArray[i].y);
					currentPlatePointList.add(platePoint);
					currentPlates.add(platesArray[i]);
					if (utils.isNewPlate(platePointList, platePoint)) {
						isHasNewPlate = true;
					}
				}

				if (platesArray.length > 0) {
					platePointList.clear();
					platePointList.addAll(currentPlatePointList);
				} else {
					platePointList.clear();
				}

				// If isHasNewPlate --> get sub images (ROI) --> Add to Adapter
				// (from
				// currentPlates)
				if ((isHasNewPlate || isFail) && !isRunningTask) {
					Log.e(TAG, "START DoOCR task!!!!");
					//new DoOCR(currentPlates, mRgba, this).execute(); // Tuan 3/5/2015 commented
				}
			}
		}

		public void updateResult(String result) {
			// TODO Auto-generated method stub

			foundNumberPlate.setText(result);

		}

	}

	public void updateResult(String result) {
		// TODO Auto-generated method stub

		foundNumberPlate.setText(result);
	}



}
