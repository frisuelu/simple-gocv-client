// What it does:
//
// This example uses a deep neural network to perform object detection.
// It can be used with either the Caffe face tracking or Tensorflow object detection models that are
// included with OpenCV 3.4
//
// To perform face tracking with the Caffe model:
//
// Download the model file from:
// https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/res10_300x300_ssd_iter_140000.caffemodel
//
// You will also need the prototxt config file:
// https://raw.githubusercontent.com/opencv/opencv/master/samples/dnn/face_detector/deploy.prototxt
//
// To perform object tracking with the Tensorflow model:
//
// Download and extract the model file named "frozen_inference_graph.pb" from:
// http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v1_coco_2017_11_17.tar.gz
//
// You will also need the pbtxt config file:
// https://gist.githubusercontent.com/dkurt/45118a9c57c38677b65d6953ae62924a/raw/b0edd9e8c992c25fe1c804e77b06d20a89064871/ssd_mobilenet_v1_coco_2017_11_17.pbtxt
//
// How to run:
//
// 		go run ./cmd/dnn-detection/main.go [videosource] [modelfile] [configfile] ([backend] [device])
//
// +build example

package main

import (
	"fmt"
	"image"
	"image/color"
	"os"
	"path/filepath"

	"gocv.io/x/gocv"
)

func main() {
	if len(os.Args) < 4 {
		fmt.Println("How to run:\ndnn-detection [videosource] [modelfile] [configfile] ([backend] [device])")
		return
	}

	// parse args
	deviceID := os.Args[1]
	model := os.Args[2]
	config := os.Args[3]
	backend := gocv.NetBackendDefault
	if len(os.Args) > 4 {
		backend = gocv.ParseNetBackend(os.Args[4])
	}

	target := gocv.NetTargetCPU
	if len(os.Args) > 5 {
		target = gocv.ParseNetTarget(os.Args[5])
	}

	// open capture device
	webcam, err := gocv.OpenVideoCapture(deviceID)
	if err != nil {
		fmt.Printf("Error opening video capture device: %v\n", deviceID)
		return
	}
	defer webcam.Close()

	window := gocv.NewWindow("DNN Detection")
	defer window.Close()

	img := gocv.NewMat()
	defer img.Close()

	// open DNN object tracking model
	net := gocv.ReadNet(model, config)
	if net.Empty() {
		fmt.Printf("Error reading network model from : %v %v\n", model, config)
		return
	}
	defer net.Close()
	net.SetPreferableBackend(gocv.NetBackendType(backend))
	net.SetPreferableTarget(gocv.NetTargetType(target))

	var ratio float64
	var mean gocv.Scalar
	var swapRGB bool

	if filepath.Ext(model) == ".caffemodel" {
		ratio = 1.0
		mean = gocv.NewScalar(104, 177, 123, 0)
		swapRGB = false
	} else {
		ratio = 1.0 / 127.5
		mean = gocv.NewScalar(127.5, 127.5, 127.5, 0)
		swapRGB = true
	}

	fmt.Printf("Start reading device: %v\n", deviceID)

	for {
		if ok := webcam.Read(&img); !ok {
			fmt.Printf("Device closed: %v\n", deviceID)
			return
		}
		if img.Empty() {
			continue
		}

		// convert image Mat to 300x300 blob that the object detector can analyze
		blob := gocv.BlobFromImage(img, ratio, image.Pt(300, 300), mean, swapRGB, false)

		// feed the blob into the detector
		net.SetInput(blob, "")

		// run a forward pass thru the network
		prob := net.Forward("")

		performDetection(&img, prob)

		prob.Close()
		blob.Close()

		window.IMShow(img)
		if window.WaitKey(1) >= 0 {
			break
		}
	}
}

// performDetection analyzes the results from the detector network,
// which produces an output blob with a shape 1x1xNx7
// where N is the number of detections, and each detection
// is a vector of float values
// [batchId, classId, confidence, left, top, right, bottom]
func performDetection(frame *gocv.Mat, results gocv.Mat) {

	/*data, ip := ioutil.ReadFile("labels.pbtxt")

	if ip != nil {
		log.Fatal(ip)
	}

	fmt.Println(string(data))*/

	type Items struct {
		Name         string
		ID           int
		Display_name string
	}

	labels := map[int][]string{}

	labels[1] = append(labels[1], "person")
	labels[2] = append(labels[2], "bycicle")
	labels[3] = append(labels[3], "car")
	labels[4] = append(labels[4], "motorcycle")
	labels[5] = append(labels[5], "airplane")
	labels[6] = append(labels[6], "bus")
	labels[7] = append(labels[7], "train")
	labels[8] = append(labels[8], "truck")
	labels[9] = append(labels[9], "boat")
	labels[10] = append(labels[10], "traffic light")
	labels[11] = append(labels[11], "fire hydrant")
	labels[12] = append(labels[12], "stop sign")
	labels[13] = append(labels[13], "parking meter")
	labels[14] = append(labels[14], "bench")
	labels[15] = append(labels[15], "bird")
	labels[16] = append(labels[16], "cat")
	labels[17] = append(labels[17], "dog")
	labels[18] = append(labels[18], "horse")
	labels[19] = append(labels[19], "sheep")
	labels[20] = append(labels[20], "cow")
	labels[21] = append(labels[21], "elephant")
	labels[22] = append(labels[22], "bear")
	labels[23] = append(labels[23], "zebra")
	labels[24] = append(labels[24], "giraffe")
	labels[25] = append(labels[25], "backpack")
	labels[26] = append(labels[26], "umbrella")
	labels[27] = append(labels[27], "handbag")
	labels[28] = append(labels[28], "tie")
	labels[29] = append(labels[29], "suitcase")
	labels[30] = append(labels[30], "frisbee")
	labels[31] = append(labels[31], "skis")
	labels[32] = append(labels[32], "snowboard")
	labels[33] = append(labels[33], "sports ball")
	labels[34] = append(labels[34], "kite")
	labels[35] = append(labels[35], "baseball bat")
	labels[36] = append(labels[36], "baseball glove")
	labels[37] = append(labels[37], "skateboard")
	labels[38] = append(labels[38], "surfboard")
	labels[39] = append(labels[39], "tennis racket")
	labels[40] = append(labels[40], "bottle")
	labels[41] = append(labels[41], "wine glass")
	labels[42] = append(labels[42], "cup")
	labels[43] = append(labels[43], "fork")
	labels[44] = append(labels[44], "knife")
	labels[45] = append(labels[45], "spoon")
	labels[46] = append(labels[46], "bowl")
	labels[47] = append(labels[47], "banana")
	labels[48] = append(labels[48], "apple")
	labels[49] = append(labels[49], "sandwich")
	labels[50] = append(labels[50], "orange")
	labels[51] = append(labels[51], "broccoli")
	labels[52] = append(labels[52], "carrot")
	labels[53] = append(labels[53], "hot dog")
	labels[54] = append(labels[54], "pizza")
	labels[55] = append(labels[55], "donut")
	labels[56] = append(labels[56], "cake")
	labels[57] = append(labels[57], "chair")
	labels[58] = append(labels[58], "couch")
	labels[59] = append(labels[59], "potted plant")
	labels[60] = append(labels[60], "bed")
	labels[61] = append(labels[61], "dining table")
	labels[62] = append(labels[62], "toilet")
	labels[63] = append(labels[63], "tv")
	labels[64] = append(labels[64], "laptop")
	labels[65] = append(labels[65], "mouse")
	labels[66] = append(labels[66], "remote")
	labels[67] = append(labels[67], "keyboard")
	labels[68] = append(labels[68], "cell phone")
	labels[69] = append(labels[69], "microwave")
	labels[70] = append(labels[70], "oven")
	labels[71] = append(labels[71], "toaster")
	labels[72] = append(labels[72], "sink")
	labels[73] = append(labels[73], "refrigerator")
	labels[74] = append(labels[74], "book")
	labels[75] = append(labels[75], "clock")
	labels[76] = append(labels[76], "vase")
	labels[77] = append(labels[77], "scissors")
	labels[78] = append(labels[78], "teddy bear")
	labels[79] = append(labels[79], "hair drier")
	labels[80] = append(labels[80], "toothbrush")

	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		classId := results.GetFloatAt(0, i+1)

		if confidence > 0.5 {
			fmt.Printf("%q\n", labels[int(classId)])

			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))
			gocv.Rectangle(frame, image.Rect(left, top, right, bottom), color.RGBA{0, 255, 0, 0}, 2)
		}
	}
}
