// What it does:
//

/*
	Modifying file main.go from dnn-detection module to intake still images instead of video
	from the webcam or other device and run predictions on them based on the input model.
	Both Caffe and Tensorflow models are supported, but in this case a Tensorflow 1.15 one
	is used to perform object detection on cows.

	Image reading and display code is taken from the showimage module.

	The testing images must be in the same directory as this file.
*/

// 	How to run:
//
// 		go run ./cmd/dnn-detection/main.go [modelfile] [configfile] ([backend] [device])
//

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
	if len(os.Args) < 3 {
		fmt.Println("How to run:\ndnn-detection [modelfile] [configfile] ([backend] [device])")
		return
	}

	// parse args
	// set testing directory and read images; e.g., ./image.jpeg
	test_dir := os.Args[1]
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

	image_read := gocv.IMRead(test_dir, gocv.IMReadColor)
	if image_read.Empty() {
		fmt.Printf("cannot read image %s\n", test_dir)
		return
	}

	window := gocv.NewWindow("Object detection showcase")
	defer window.Close()

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

	// perform detection on images

	for {
		// convert image Mat to 300x300 blob that the object detector can analyze
		blob := gocv.BlobFromImage(image_read, ratio, image.Pt(300, 300), mean, swapRGB, false)

		// feed the blob into the detector
		net.SetInput(blob, "")

		// run a forward pass thru the network
		prob := net.Forward("")

		performDetection(&image_read, prob)

		prob.Close()
		blob.Close()

		window.IMShow(image_read)
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

	for i := 0; i < results.Total(); i += 7 {
		confidence := results.GetFloatAt(0, i+2)
		//classId := results.GetFloatAt(0, i+1)
		//fmt.Println(int(classId))
		//fmt.Println(confidence)

		if confidence > 0.2 {
			//fmt.Printf("%q\n", labels[int(classId)])

			left := int(results.GetFloatAt(0, i+3) * float32(frame.Cols()))
			top := int(results.GetFloatAt(0, i+4) * float32(frame.Rows()))
			right := int(results.GetFloatAt(0, i+5) * float32(frame.Cols()))
			bottom := int(results.GetFloatAt(0, i+6) * float32(frame.Rows()))
			fmt.Println(left, top, right, bottom)
			gocv.Rectangle(frame, image.Rect(left, top, right, bottom), color.RGBA{0, 0, 255, 0}, 1)
			gocv.Rectangle(frame, image.Rect(left+3, top, right-200, bottom-160), color.RGBA{0, 0, 255, 0}, int(gocv.Filled))
			gocv.PutText(frame, "Cow", image.Pt(left+8, top+20), gocv.FontHersheyPlain, 1.2, color.RGBA{0, 0, 0, 0}, 2)
		}
	}
}
