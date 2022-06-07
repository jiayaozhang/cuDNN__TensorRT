# TensorRT Plugin Introduction

- DPRNN Segmentation module graph

1. write plugin

2. write plugin creator 

3. register tensorRT plugin

4. serialization 

* Static shape, use IPluginV2IOExt

* Dynamic Shape, use IPluginV2DynamicExt

Static Shape Plugin API can only be used in 

1. Network definition

2. Deserialize time

3. MycustomPlugin() = delete;


Getserializationsize()

1. size_t serializationsize() const;

2. void serialize(void* buffer) const;

3. const char* getPluginType() const

4. const char* getPluginVersion() const;

Dynamic shape Plugin API

1. `getPluginRegistry()->registerCreator(*mCreator, libNamespace);`


# How to debug plugin in TensorRT

TRT是闭源软件，API相对比较复杂。

1. 无论是使用API还是parser构建网络，模型转换完后，结果误差很大，怎么办？

2. 增加了自定义plugin 实现算子合并，结果对不上，怎么办？

3. 使用FP16 or INT8优化策略后，算法精确度掉了很多，怎么办？