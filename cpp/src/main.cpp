#include <algorithm>
#include <cmath>
#include <iostream>
#include <string>
#include <vector>
#include <opencv2/opencv.hpp>
#include <openvino/openvino.hpp>

// Same order as in Python (sorted A-Z) -> index 0..5
const std::vector<std::string> CLASSES = {
    "crazing", "inclusion", "patches", "pitted_surface", "rolled-in_scale", "scratches"};

// Prepare one image exactly like dataset.py (val/test transforms)
std::vector<float> preprocess(const std::string& path) {
    cv::Mat img = cv::imread(path, cv::IMREAD_COLOR);       // OpenCV loads as BGR
    if (img.empty()) {
        throw std::runtime_error("cannot read image: " + path);
    }
    cv::cvtColor(img, img, cv::COLOR_BGR2RGB);               // -> RGB, like PIL
    cv::resize(img, img, cv::Size(224, 224));                // Resize((224, 224))
    img.convertTo(img, CV_32FC3, 1.0 / 255.0);               // ToTensor: 0..255 -> 0..1

    const float mean[3] = {0.485f, 0.456f, 0.406f};
    const float stdv[3] = {0.229f, 0.224f, 0.225f};

    // Normalize and reorder from HWC (pixel by pixel) to CHW (channel by channel)
    std::vector<float> data(3 * 224 * 224);
    for (int c = 0; c < 3; ++c)
        for (int y = 0; y < 224; ++y)
            for (int x = 0; x < 224; ++x)
                data[c * 224 * 224 + y * 224 + x] =
                    (img.at<cv::Vec3f>(y, x)[c] - mean[c]) / stdv[c];
    return data;
}

int main(int argc, char* argv[]) {
    if (argc < 3) {
        std::cout << "usage: ./classify model.xml image.jpg\n";
        return 1;
    }

    // Load and compile the model for the CPU
    ov::Core core;
    ov::CompiledModel compiled = core.compile_model(argv[1], "CPU");
    ov::InferRequest request = compiled.create_infer_request();

    // Prepare the image and hand it to the model
    std::vector<float> data = preprocess(argv[2]);
    ov::Tensor input(ov::element::f32, ov::Shape{1, 3, 224, 224}, data.data());
    request.set_input_tensor(input);

    // Run the model
    request.infer();
    const float* scores = request.get_output_tensor().data<float>();

    // Softmax: turn the 6 scores into percentages
    float max_score = *std::max_element(scores, scores + 6);
    std::vector<float> probs(6);
    float sum = 0.0f;
    for (int i = 0; i < 6; ++i) {
        probs[i] = std::exp(scores[i] - max_score);
        sum += probs[i];
    }
    int best = std::max_element(probs.begin(), probs.end()) - probs.begin();

    std::cout << CLASSES[best] << " (" << 100.0f * probs[best] / sum << "%)\n";
    return 0;
}