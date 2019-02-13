#include"mobilenet.h"

namespace tflite4mobile_net{
    double get_us(struct timeval t) { return (t.tv_sec * 1000000 + t.tv_usec); }

    MobileNet::MobileNet(const std::string modelfile, const std::string dictfile) : buffer(nullptr){
        TfLiteStatus load_model_status = load_model_from_file(modelfile);
        if(load_model_status == kTfLiteError)
        {
            throw std::runtime_error("load model file failed\n");
        }
        TfLiteStatus load_dict_status = load_dict(dictfile);
        if(load_dict_status == kTfLiteError)
           throw std::runtime_error("load model file failed\n");
    }

    TfLiteStatus MobileNet::load_dict(const std::string dictfile){
        int index = 0;
        std::fstream reader(dictfile, std::fstream::in);
        std::string hanzi;
        while(std::getline(reader, hanzi)){
            id2hanzi[index++] = hanzi;
        }

        /*test id2hanzi*/

        return kTfLiteOk;
    }

    TfLiteStatus MobileNet::load_model_from_file(const std::string modelfile) {
        model_ = tflite::FlatBufferModel::BuildFromFile(modelfile.c_str());
        if(!model_){
            LOG(FATAL) << "\nFailed to mmap model " << modelfile << "\n";
            throw std::runtime_error("\nFailed to mmap model");
            return kTfLiteError;
        }

        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);
        if(!interpreter_){
            LOG(FATAL) << "Failed to construct interpreter\n";
            throw std::runtime_error("Failed to construct interpreter\n");
            return kTfLiteError;
         }
        interpreter_ -> UseNNAPI(false);
        if(interpreter_ -> AllocateTensors() != kTfLiteOk)
            std::cout << "allocate tensors failed\n";
        //interpreter_ ->SetNumThreads(2);
        return kTfLiteOk;
    } 

    TfLiteStatus MobileNet::load_model_from_buffer(const std::string model_filename){

        std::ifstream filein(model_filename.c_str(), std::ios::binary | std::ios::ate);
        if(!filein){
            //throw std::runtime_error("Unable open model file\n");
            return kTfLiteError;
        }
        std::streamsize size = filein.tellg();
        filein.seekg(0, std::ios::beg);
        buffer = new char[size];
        if(!filein.read(buffer, size)){
            //std::cout << "read into buffer failed\n";
            return kTfLiteError;
        }
        filein.close();
        
        /*
        struct timeval start_time, stop_time;

        gettimeofday(&start_time, NULL);
        xor_encrypt_decrypt(key, buffer, size); 
        gettimeofday(&stop_time, NULL);
        LOG(INFO) << "decrypt time: "
                  << (get_us(stop_time) - get_us(start_time)) / (1000)
                  << " ms \n";
          */

        model_ = tflite::FlatBufferModel::BuildFromBuffer(buffer, size);
        if(!model_){
            LOG(FATAL) << "\nFailed to mmap model " << model_filename << "\n";
            //throw std::runtime_error("\nFailed to mmap model");
            return kTfLiteError;
        }

        tflite::InterpreterBuilder(*model_, resolver_)(&interpreter_);
        if(!interpreter_){
            LOG(FATAL) << "Failed to construct interpreter\n";
            //throw std::runtime_error("Failed to construct interpreter\n");
            return kTfLiteError;
         }
        interpreter_ -> UseNNAPI(false);
        return kTfLiteOk;
    }

	std::vector<size_t> MobileNet::argmax(const std::vector<float> &v) {

	  // initialize original index locations
        std::vector<size_t> idx(v.size());
        std::iota(idx.begin(), idx.end(), 0);

		// sort indexes based on comparing values in v
        std::sort(idx.begin(), idx.end(),[&v](size_t i1, size_t i2) {return v[i1] > v[i2];});
	    return idx;
    }


    std::vector<std::pair<std::string, float>> MobileNet::predict(uint8_t *img, int topk){

        //std::cout << "topk:" << topk << "\n";
        std::vector<std::pair<std::string, float>>  hanzi;
        const int image_height=64;
        const int image_width=64;
        const int channel=1;
        const int dict_size=3755;

        /*
        int input_tensor_ix  = interpreter_ ->inputs()[0];
        auto type = interpreter_ ->tensor(input_tensor_ix)->type;
        std::cout << "input tensor index: " << input_tensor_ix << "\n";
        std::cout << "input tensor type: " << type << "\n";

        auto input = interpreter_ ->typed_tensor<float>(input_tensor_ix);
        std::cout << "input(0) name: " << interpreter_ ->GetInputName(0) << "\n";

        TfLiteIntArray* dims = interpreter_ ->tensor(88)->dims;
        int batch_size = dims->data[0];
        int wanted_height = dims->data[1];
        int wanted_width = dims->data[2];
        int wanted_channels = dims->data[3];

        std::cout << batch_size << " " <<  wanted_height << " " << wanted_width << " " << wanted_channels << "\n"; 
        int output_tensor_ix  = interpreter_ ->outputs()[0];
        auto out_type = interpreter_ ->tensor(output_tensor_ix)->type;
        std::cout << "output tensor type: " << out_type << "\n";

        */

        uint8_t* input = interpreter_ ->typed_input_tensor<uint8_t>(0);

        struct timeval start_time, stop_time;

        int num_pixels = image_width*image_height*channel;
        memcpy(input, img, sizeof(uint8_t)*num_pixels);

       // gettimeofday(&start_time, NULL);
        if(interpreter_ ->Invoke() != kTfLiteOk){
            std::cout << "invoke failed\n";
        }
       // gettimeofday(&stop_time, NULL);

        //std::cout << "Invoke cost: " << (get_us(stop_time) - get_us(start_time))/1000.0 << "ms\n";
        auto probs = interpreter_ ->typed_output_tensor<float>(0);
        std::vector<float>v{probs, probs + dict_size};
        //for(int i = 0; i < 3756; i++)
         //   std::cout << probs[i];

        //struct timeval start_time, stop_time;
       // gettimeofday(&start_time, NULL);
        std::vector<size_t> ix = argmax(v);
      //  gettimeofday(&stop_time, NULL);
      //  std::cout << "Argmax cost: " << (get_us(stop_time) - get_us(start_time))/1000.0 << "ms\n";
        //std::cout << "ix size:" << ix.size();
        for(int i = 0; i < ix.size() && i < topk; ++i){
            //std::cout << id2hanzi[ix[i]] << std::endl;
            hanzi.push_back(std::make_pair(id2hanzi[ix[i]],v[ix[i]]));
        }
        return hanzi;
    }
}
