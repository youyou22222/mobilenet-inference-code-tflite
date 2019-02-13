#include"mobilenet.h"
#include"mobilenet_wrapper.h"
#include<string>
#include<utility>
#include<vector>

MobileNetWrapper *CreateInstance(const char* model_file_name, const char *dictfile){
    MobileNetWrapper *p = nullptr;
    try {
        p = new MobileNetWrapper(model_file_name, dictfile);
    }
    catch (const std::runtime_error& error){
        p = nullptr;
    }
    return p;
}

MobileNetWrapper::MobileNetWrapper(const char* model_file_name, const char *dictfile){
    std::string modelfile(model_file_name);
    std::string dict(dictfile);
    pMobile = new tflite4mobile_net::MobileNet(modelfile, dict);
}

struct hanzi_prob *MobileNetWrapper::predict(uint8_t *img, int topk){
   if(pMobile != nullptr) {
        tflite4mobile_net::MobileNet *cnn = static_cast<tflite4mobile_net::MobileNet*> (pMobile);

        std::vector<std::pair<std::string, float>> res = cnn ->predict(img, topk);
        if(res.size() <= 0)
            return nullptr;
        std::string hanzi = "";
        size_t n = res.size();
        struct hanzi_prob *zh_prob = new hanzi_prob;//(nullptr, nullptr);
        zh_prob ->prob = new float[n];
        for(int i = 0; i < n; ++i){
            hanzi += res[i].first;
            hanzi += " ";
            zh_prob ->prob[i] = res[i].second;
        }

        size_t hz_len = hanzi.size();
        zh_prob ->hanzi = new char [hz_len];
        strncpy(zh_prob ->hanzi, hanzi.c_str(),hz_len); 

        zh_prob ->hanzi[hanzi.size()-1] = '\0';
        return zh_prob;
   }
   return nullptr;
}

MobileNetWrapper::~MobileNetWrapper(){
    if (pMobile != nullptr){
        tflite4mobile_net::MobileNet *p = static_cast<tflite4mobile_net::MobileNet*> (pMobile);
        delete p;
    }
}
