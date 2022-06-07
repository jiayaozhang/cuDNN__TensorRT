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

2. `how to debug plugin in TensorRT`