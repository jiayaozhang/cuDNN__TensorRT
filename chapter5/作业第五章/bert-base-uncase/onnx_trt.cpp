#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"

#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include "NvOnnxParser.h"

using namespace nvonnxparser;
using samplesCommon::SampleUniquePtr;

#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>

//class MyLogger : public ILogger
//{
//    void log(Severity severity, const char* msg) override
//    {
//        // suppress info-level messages
//        if (severity <= Severity::kWARNING)
//            std::cout << msg << std::endl;
//    }
//};

int main(int argc, char** argv)
{
    //std::string modelFile = "D:/Data/2021_11_10_MarkerLocalization/Training_CodedSuit/MarkerLocalizationNetSimplierNet_V3_Stride8/Save/savedpb/model.pb.onnx";
    std::string modelFile = "G:/TMP/model-sim.onnx";
    std::string clusteringInfoFile = "C:/Code/01_Vision\MarkerTracking/S03_MarkerTracker/Outputs/S09_AnalyzeCrops/ClusterInfo.json";
    int32_t dlaCore{ -1 };               //!< Specify the DLA core to run network on.

    samplesCommon::Args args;
    bool argsOK = samplesCommon::parseArgs(args, argc, argv);
    if (!argsOK)
    {
        sample::gLogError << "Invalid arguments" << std::endl;
        return EXIT_FAILURE;
    }
    if (args.help)
    {
        return EXIT_SUCCESS;
    }


    IBuilder* builder = createInferBuilder(sample::gLogger.getTRTLogger());
    IBuilderConfig* config = builder->createBuilderConfig();

    uint32_t explicitBatchflag = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);

    INetworkDefinition * network = builder->createNetworkV2(explicitBatchflag);

    IParser* parser = createParser(*network, sample::gLogger.getTRTLogger());
    auto parsed = parser->parseFromFile(modelFile.c_str(), static_cast<int>(sample::gLogger.getReportableSeverity()));
    for (int32_t i = 0; i < parser->getNbErrors(); ++i)
    {
        std::cout << parser->getError(i)->desc() << std::endl;
    }

    auto numInputs = network->getNbInputs();
    std::cout << "Number of inputs: " << numInputs << "\n";

    auto input0 = network->getInput(0);
    std::cout << "Input Name: " << input0->getName() << ", Input Demisions:" << input0->getDimensions() << "\n";
    auto output = network->getOutput(0);
    std::cout << "Output Name: " << output->getName() << ". Output Demisions:" << output->getDimensions() << "\n";

    config->setMaxWorkspaceSize(8_GiB);
    //samplesCommon::enableDLA(builder, config, dlaCore);

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    config->setProfileStream(*profileStream);

    auto profile = builder->createOptimizationProfile();
    std::string inputName = "inputs:0";
    const int calibBatchSize{ 1 };
    // We do not need to check the return of setDimension and setCalibrationProfile here as all dims are explicitly set
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMIN, Dims4{ calibBatchSize, 1184, 1712,  3 });
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kOPT, Dims4{ calibBatchSize, 1184, 1712,  3 });
    profile->setDimensions(inputName.c_str(), OptProfileSelector::kMAX, Dims4{ calibBatchSize, 1184, 1712,  3});
    config->addOptimizationProfile(profile);

    SampleUniquePtr<IHostMemory> plan{ builder->buildSerializedNetwork(*network, *config) };
    SampleUniquePtr<IRuntime> runtime{ createInferRuntime(sample::gLogger.getTRTLogger()) };
    SampleUniquePtr<nvinfer1::ICudaEngine> predictEngine{runtime->deserializeCudaEngine(plan->data(), plan->size())};


}