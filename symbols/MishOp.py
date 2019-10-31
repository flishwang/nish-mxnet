import mxnet as mx

act_type_replace='mish'
nr = 2.0/3.0
mx.sym.raw_activation=mx.sym.Activation

def Act(data,act_type,name=None,**kwargs):
    if act_type=='relu':
        if  act_type_replace =='nish':
            expx=mx.sym.exp(data*(1.0/nr))
            return mx.sym.where(data>=0,data, expx* (-2.5*nr + 4 *nr* expx - 1.5 *nr* expx * expx))
        if  act_type_replace =='pish':
            expx=mx.sym.exp(data*(1.0/nr))
            return mx.sym.where(data>=0,data,
                                expx* (-13./3.*nr + expx * (9.5 *nr + expx *( -7 *nr + 11./6. * nr * expx))))
        if act_type_replace == 'mish':
            return data * mx.sym.tanh(mx.sym.raw_activation(data,'softrelu'))
        if act_type_replace == 'lish':
            return mx.sym.where(data>=0,data,5.0/3.0*data * mx.sym.tanh(mx.sym.raw_activation(data,'softrelu')))
        return mx.sym.Custom(data,op_type='mish_rtc',act_type=act_type_replace,name=name,**kwargs)
    else:
        return mx.sym.raw_activation(data,act_type,name,**kwargs)

mx.sym.Activation=Act