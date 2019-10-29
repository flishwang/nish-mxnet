import mxnet as mx

class base_operator(mx.operator.CustomOp):
    def __init__(self,shape):
        super(base_operator, self).__init__()
        self.shape = shape
        total = 1
        for s in shape:
            total *= s
        threads = 1
        while total % 2 == 0 and threads < 256:
            threads *= 2
            total /= 2
        b3 = 1

        while total % 2 == 0 and b3 < 256:
            b3 *= 2
            total /= 2
        while total % 3 == 0 and b3 < 256:
            b3 *= 3
            total /= 3
        b2 = 1

        while total % 2 == 0 and b2 < 4096 and b2 < total:
            b2 *= 2
            total /= 2
        while total % 3 == 0 and b2 < 4096 and b2 < total:
            b2 *= 3
            total /= 3
        while total % 5 == 0 and b2 < 4096 and b2 < total:
            b2 *= 5
            total /= 5
        left = total
        self.kernelShape = (left, b2, b3, threads, 1, 1)




class MishRTCOperator(base_operator):
    source_fwd = r'''
    template<typename DType>
    __global__ void mishpy_fwd(DType *x){
    	int i = (blockIdx.x * gridDim.y + blockIdx.y) * (blockDim.x * gridDim.z) + (blockIdx.z * blockDim.x + threadIdx.x);
    	x[i] = x[i] * tanhf(log(1 + exp(x[i])));
    }
    '''

    source_bwd = r'''
    template<typename DType>
    __global__ void mishpy_bwd(DType *x){
    	int i = (blockIdx.x * gridDim.y + blockIdx.y) * (blockDim.x * gridDim.z) + (blockIdx.z * blockDim.x + threadIdx.x);
    	float exp_x = exp(x[i]);
    	float tanhf_softplus_x = tanhf(log(1 + exp_x));
    	x[i] = tanhf_softplus_x + x[i] * (1 - tanhf_softplus_x*tanhf_softplus_x) * (exp_x/(1 + exp_x));
    	x[i] = isnan(x[i])?1.0:x[i];
    }
    '''
    module_fwd = mx.rtc.CudaModule(source_fwd, exports='mishpy_fwd<float>')
    module_bwd = mx.rtc.CudaModule(source_bwd, exports='mishpy_bwd<float>')
    func32_fwd = module_fwd.get_kernel("mishpy_fwd<float>", "float *x")
    func32_bwd = module_bwd.get_kernel("mishpy_bwd<float>", "float *x")

    def __init__(self,data_shape):
        super(MishRTCOperator, self).__init__(data_shape)

    def forward(self, is_train, req, in_data, out_data, aux):
        y = in_data[0].astype('float32')
        MishRTCOperator.func32_fwd.launch([y], y.context, self.kernelShape[0:3],self.kernelShape[3:6])

        self.assign(out_data[0], req[0], y.astype(in_data[0].dtype))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = in_data[0].astype('float32')
        MishRTCOperator.func32_bwd.launch([y], y.context, self.kernelShape[0:3],self.kernelShape[3:6])
        self.assign(in_grad[0], req[0], y.astype(in_data[0].dtype) * out_grad[0])

class NishRTCOperator(base_operator):
    source_fwd = r'''
    template<typename DType>
    __global__ void nishpy_fwd(DType *x){
    	int i = (blockIdx.x * gridDim.y + blockIdx.y) * (blockDim.x * gridDim.z) + (blockIdx.z * blockDim.x + threadIdx.x);
        DType expx= exp(x[i]);
        x[i] = x[i]>=0 ? x[i] : -2.5 * expx + 4 * expx * expx - 1.5 * expx * expx * expx;
    }
    '''

    source_bwd = r'''
    template<typename DType>
    __global__ void nishpy_bwd(DType *x){
    	int i = (blockIdx.x * gridDim.y + blockIdx.y) * (blockDim.x * gridDim.z) + (blockIdx.z * blockDim.x + threadIdx.x);
    	DType expx= exp(x[i]);
    	x[i] = x[i]>=0? 1 : -2.5 * expx + 8 * expx * expx - 4.5 * expx * expx * expx;
    }
    '''
    module_fwd = mx.rtc.CudaModule(source_fwd, exports='nishpy_fwd<float>')
    module_bwd = mx.rtc.CudaModule(source_bwd, exports='nishpy_bwd<float>')
    func32_fwd = module_fwd.get_kernel("nishpy_fwd<float>", "float *x")
    func32_bwd = module_bwd.get_kernel("nishpy_bwd<float>", "float *x")

    def __init__(self,data_shape):
        super(NishRTCOperator, self).__init__(data_shape)

    def forward(self, is_train, req, in_data, out_data, aux):
        y = in_data[0].astype('float32')
        NishRTCOperator.func32_fwd.launch([y], y.context, self.kernelShape[0:3],self.kernelShape[3:6])

        self.assign(out_data[0], req[0], y.astype(in_data[0].dtype))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = in_data[0].astype('float32')
        NishRTCOperator.func32_bwd.launch([y], y.context, self.kernelShape[0:3],self.kernelShape[3:6])
        self.assign(in_grad[0], req[0], y.astype(in_data[0].dtype) * out_grad[0])

class HishRTCOperator(base_operator):
    source_fwd = r'''
    template<typename DType>
    __global__ void hishpy_fwd(DType *x){
    	int i = (blockIdx.x * gridDim.y + blockIdx.y) * (blockDim.x * gridDim.z) + (blockIdx.z * blockDim.x + threadIdx.x);
        x[i] = (x[i]>=1.0 || x[i]<=-1.0) ? x[i] : 0.5 + 0.5 * x[i] *x[i];
    }
    '''

    source_bwd = r'''
    template<typename DType>
    __global__ void hishpy_bwd(DType *x){
    	int i = (blockIdx.x * gridDim.y + blockIdx.y) * (blockDim.x * gridDim.z) + (blockIdx.z * blockDim.x + threadIdx.x);
    	x[i] = x[i]>=1.0 ? 1.0 : (x[i]<=-1.0 ? -1.0 : x[i]);
    }
    '''
    module_fwd = mx.rtc.CudaModule(source_fwd, exports='hishpy_fwd<float>')
    module_bwd = mx.rtc.CudaModule(source_bwd, exports='hishpy_bwd<float>')
    func32_fwd = module_fwd.get_kernel("hishpy_fwd<float>", "float *x")
    func32_bwd = module_bwd.get_kernel("hishpy_bwd<float>", "float *x")

    def __init__(self,data_shape):
        super(HishRTCOperator, self).__init__(data_shape)

    def forward(self, is_train, req, in_data, out_data, aux):
        y = in_data[0].astype('float32')
        HishRTCOperator.func32_fwd.launch([y], y.context, self.kernelShape[0:3],self.kernelShape[3:6])

        self.assign(out_data[0], req[0], y.astype(in_data[0].dtype))

    def backward(self, req, out_grad, in_data, out_data, in_grad, aux):
        y = in_data[0].astype('float32')
        HishRTCOperator.func32_bwd.launch([y], y.context, self.kernelShape[0:3],self.kernelShape[3:6])
        self.assign(in_grad[0], req[0], y.astype(in_data[0].dtype) * out_grad[0])



@mx.operator.register('mish_rtc')
class NishRTCProp(mx.operator.CustomOpProp):
    def __init__(self,act_type='mish'):
        super(NishRTCProp, self).__init__(need_top_grad=True)
        self.act_type=act_type

    def list_arguments(self):
        return ['data']

    def list_outputs(self):
        return ['output']

    def infer_shape(self, in_shape):
        data_shape = in_shape[0]
        out_shape = data_shape
        return [data_shape], [out_shape]

    def create_operator(self, ctx, shapes, dtypes):
        if self.act_type=='mish':
            return MishRTCOperator(shapes[0])
        elif self.act_type=='nish':
            return NishRTCOperator(shapes[0])
        elif self.act_type=='hish':
            return HishRTCOperator(shapes[0])

act_type_replace='mish'
mx.sym.raw_activation=mx.sym.Activation

def Act(data,act_type,name=None,**kwargs):
    if act_type=='relu':
        return mx.sym.Custom(data,op_type='mish_rtc',act_type=act_type_replace,name=name,**kwargs)
    else:
        return mx.sym.raw_activation(data,act_type,name,**kwargs)

mx.sym.Activation=Act
