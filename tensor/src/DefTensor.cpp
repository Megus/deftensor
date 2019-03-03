// myextension.cpp
// Extension lib defines
#define LIB_NAME "DefTensor"
#define MODULE_NAME "DefTensor"

// include the Defold SDK
#include <dmsdk/sdk.h>

#if defined(DM_PLATFORM_IOS) || defined(DM_PLATFORM_OSX) || defined(DM_PLATFORM_ANDROID)

#include "tensorflow/lite/kernels/register.h"
#include "tensorflow/lite/model.h"
#include "tensorflow/lite/op_resolver.h"
#include "tensorflow/lite/string_util.h"

std::unique_ptr<tflite::FlatBufferModel> model;
tflite::ops::builtin::BuiltinOpResolver resolver;
std::unique_ptr<tflite::Interpreter> interpreter;
TfLiteDelegate* delegate;

#define TFLITE_SUPPORTED

#endif

static void pushFloatArray(lua_State* L, float* array, int size) {
    lua_createtable(L, size, 0);
    for (int i = 0; i < size; i++) {
        lua_pushnumber(L, (double)(array[i]));
        lua_rawseti(L, -2, i + 1);
    }
}

static int loadModel(lua_State* L) {
    // Load buffer with model
    dmScript::LuaHBuffer *buffer = dmScript::CheckBuffer(L, 1);
    char* bytes = 0x0;
    uint32_t size = 0;
    DM_LUA_STACK_CHECK(L, 1);
    if (buffer != NULL) {
        dmBuffer::Result r = dmBuffer::GetBytes(buffer->m_Buffer, (void**)&bytes, &size);
        // Now we have a buffer, let's load it to TFLite
#ifdef TFLITE_SUPPORTED
        model = tflite::FlatBufferModel::BuildFromBuffer(bytes, size);
        if (!model) {
            return DM_LUA_ERROR("Can't load model");
        }
        model->error_reporter();
        tflite::InterpreterBuilder(*model, resolver)(&interpreter);
        if (!interpreter) {
            return DM_LUA_ERROR("Failed to construct interpreter");
        }
        if (interpreter->AllocateTensors() != kTfLiteOk) {
            return DM_LUA_ERROR("Failed to allocate tensors!");
        }
#endif
        lua_pushinteger(L, 1);
    } else {
        return DM_LUA_ERROR("You should pass a buffer to load_model");
    }
    return 1;
}

static int runModel(lua_State* L) {
    DM_LUA_STACK_CHECK(L, 1);
#ifdef TFLITE_SUPPORTED
    //printf("1\n");
    std::vector<int> allInputs = interpreter->inputs();
    //printf("2\n");

    int input = interpreter->inputs()[0];
    TfLiteTensor *input_tensor = interpreter->tensor(input);

    bool is_quantized;
    switch (input_tensor->type) {
    case kTfLiteFloat32:
        is_quantized = false;
        break;
    case kTfLiteUInt8:
        is_quantized = true;
        break;
    default:
        return DM_LUA_ERROR("Input data type is not supported by this extension");
    }

    int luaInputSize = lua_objlen(L, 1);

    if (is_quantized) {
        // Read integer data
        uint8_t* tout = interpreter->typed_tensor<uint8_t>(input);
        for (int i = 0; i < luaInputSize; i++) {
            lua_rawgeti(L, 1, i + 1);
            tout[i] = lua_tointeger(L, 1);
            lua_pop(L, 1);
        }
    } else {
        // Read float data
        float* tout = interpreter->typed_tensor<float>(input);
        for (int i = 0; i < luaInputSize; i++) {
            lua_rawgeti(L, 1, i + 1);
            float invalue = (float)(lua_tonumber(L, -1));
            tout[i] = invalue;
            lua_pop(L, 1);
        }
    }

    if (interpreter->Invoke() != kTfLiteOk) {
        return DM_LUA_ERROR("Failed to invoke interpreter!");
    }

    // read output size from the output sensor
    const int output_tensor_index = interpreter->outputs()[0];
    TfLiteTensor* output_tensor = interpreter->tensor(output_tensor_index);
    TfLiteIntArray* output_dims = output_tensor->dims;
    if (output_dims->size != 2 || output_dims->data[0] != 1) {
        DM_LUA_ERROR("Output of the model is in invalid format.");
    }
    const int output_size = output_dims->data[1];

    std::vector<std::pair<float, int> > top_results;

    if (is_quantized) {
        uint8_t* quantized_output = interpreter->typed_output_tensor<uint8_t>(0);
        int32_t zero_point = input_tensor->params.zero_point;
        float scale = input_tensor->params.scale;
        float output[output_size];
        for (int i = 0; i < output_size; ++i) {
            output[i] = (quantized_output[i] - zero_point) * scale;
        }
        pushFloatArray(L, output, output_size);
    } else {
        float* output = interpreter->typed_output_tensor<float>(0);
        pushFloatArray(L, output, output_size);
    }

#else
    lua_pushnil(L);
#endif
    return 1;
}

// Functions exposed to Lua
static const luaL_reg Module_methods[] = {
    {"load_model", loadModel},
    {"run_model", runModel},
    {0, 0}
};

static void LuaInit(lua_State* L)
{
    int top = lua_gettop(L);

    // Register lua names
    luaL_register(L, MODULE_NAME, Module_methods);

    lua_pop(L, 1);
    assert(top == lua_gettop(L));
}

dmExtension::Result AppInitializeDefTensor(dmExtension::AppParams* params)
{
    return dmExtension::RESULT_OK;
}

dmExtension::Result InitializeDefTensor(dmExtension::Params* params)
{
    // Init Lua
    LuaInit(params->m_L);
    printf("Registered %s Extension\n", MODULE_NAME);
    return dmExtension::RESULT_OK;
}

dmExtension::Result AppFinalizeDefTensor(dmExtension::AppParams* params)
{
    return dmExtension::RESULT_OK;
}

dmExtension::Result FinalizeDefTensor(dmExtension::Params* params)
{
    return dmExtension::RESULT_OK;
}


// Defold SDK uses a macro for setting up extension entry points:
//
// DM_DECLARE_EXTENSION(symbol, name, app_init, app_final, init, update, on_event, final)

DM_DECLARE_EXTENSION(DefTensor, LIB_NAME, AppInitializeDefTensor, AppFinalizeDefTensor, InitializeDefTensor, 0, 0, FinalizeDefTensor)
