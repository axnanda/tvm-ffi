// File: src/add_one_cpu.cc
TVM_FFI_DLL_EXPORT int __tvm_ffi_add_one_cpu(void* handle, const TVMFFIAny* args,
                                             int32_t num_args, TVMFFIAny* result) {
  // Step 1. Extract inputs from `Any`
  // Step 1.1. Extract `x := args[0]`
  DLTensor* x;
  if (args[0].type_index == kTVMFFIDLTensorPtr) x = (DLTensor*)(args[0].v_ptr);
  else if (args[0].type_index == kTVMFFITensor) x = (DLTensor*)(args[0].v_c_str + sizeof(TVMFFIObject));
  else { TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor input"); return -1; }
  // Step 1.2. Extract `y := args[1]`
  DLTensor* y;
  if (args[1].type_index == kTVMFFIDLTensorPtr) y = (DLTensor*)(args[1].v_ptr);
  else if (args[1].type_index == kTVMFFITensor) y = (DLTensor*)(args[1].v_c_str + sizeof(TVMFFIObject));
  else { TVMFFIErrorSetRaisedFromCStr("ValueError", "Expects a Tensor output"); return -1; }

  // Step 2. Perform add one: y = x + 1
  for (int64_t i = 0; i < x->shape[0]; ++i) {
    ((float*)y->data)[i] = ((float*)x->data)[i] + 1.0f;
  }

  // Step 3. Return error code 0 (success)
  //
  // Note that `result` is not set, as the output is passed in via `y` argument,
  // which is functionally similar to a Python function with signature:
  //
  //   def add_one(x: Tensor, y: Tensor) -> None: ...
  return 0;
}
