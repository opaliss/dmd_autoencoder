��
��
B
AssignVariableOp
resource
value"dtype"
dtypetype�
~
BiasAdd

value"T	
bias"T
output"T" 
Ttype:
2	"-
data_formatstringNHWC:
NHWCNCHW
8
Const
output"dtype"
valuetensor"
dtypetype
;
Elu
features"T
activations"T"
Ttype:
2
.
Identity

input"T
output"T"	
Ttype
q
MatMul
a"T
b"T
product"T"
transpose_abool( "
transpose_bbool( "
Ttype:

2	
e
MergeV2Checkpoints
checkpoint_prefixes
destination_prefix"
delete_old_dirsbool(�

NoOp
M
Pack
values"T*N
output"T"
Nint(0"	
Ttype"
axisint 
C
Placeholder
output"dtype"
dtypetype"
shapeshape:
@
ReadVariableOp
resource
value"dtype"
dtypetype�
o
	RestoreV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
l
SaveV2

prefix
tensor_names
shape_and_slices
tensors2dtypes"
dtypes
list(type)(0�
?
Select
	condition

t"T
e"T
output"T"	
Ttype
H
ShardedFilename
basename	
shard

num_shards
filename
�
StatefulPartitionedCall
args2Tin
output2Tout"
Tin
list(type)("
Tout
list(type)("	
ffunc"
configstring "
config_protostring "
executor_typestring �
@
StaticRegexFullMatch	
input

output
"
patternstring
N

StringJoin
inputs*N

output"
Nint(0"
	separatorstring 
�
VarHandleOp
resource"
	containerstring "
shared_namestring "
dtypetype"
shapeshape"#
allowed_deviceslist(string)
 �"serve*2.4.02v2.4.0-rc4-71-g582c8d236cb8��
t
enc_0/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_nameenc_0/kernel
m
 enc_0/kernel/Read/ReadVariableOpReadVariableOpenc_0/kernel*
_output_shapes

:P*
dtype0
l

enc_0/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_name
enc_0/bias
e
enc_0/bias/Read/ReadVariableOpReadVariableOp
enc_0/bias*
_output_shapes
:P*
dtype0
t
enc_1/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*
shared_nameenc_1/kernel
m
 enc_1/kernel/Read/ReadVariableOpReadVariableOpenc_1/kernel*
_output_shapes

:PP*
dtype0
l

enc_1/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_name
enc_1/bias
e
enc_1/bias/Read/ReadVariableOpReadVariableOp
enc_1/bias*
_output_shapes
:P*
dtype0
t
enc_2/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:PP*
shared_nameenc_2/kernel
m
 enc_2/kernel/Read/ReadVariableOpReadVariableOpenc_2/kernel*
_output_shapes

:PP*
dtype0
l

enc_2/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:P*
shared_name
enc_2/bias
e
enc_2/bias/Read/ReadVariableOpReadVariableOp
enc_2/bias*
_output_shapes
:P*
dtype0
x
enc_out/kernelVarHandleOp*
_output_shapes
: *
dtype0*
shape
:P*
shared_nameenc_out/kernel
q
"enc_out/kernel/Read/ReadVariableOpReadVariableOpenc_out/kernel*
_output_shapes

:P*
dtype0
p
enc_out/biasVarHandleOp*
_output_shapes
: *
dtype0*
shape:*
shared_nameenc_out/bias
i
 enc_out/bias/Read/ReadVariableOpReadVariableOpenc_out/bias*
_output_shapes
:*
dtype0

NoOpNoOp
�
ConstConst"/device:CPU:0*
_output_shapes
: *
dtype0*�
value�B� B�
�
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
h


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
h

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
8

0
1
2
3
4
5
6
7
 
8

0
1
2
3
4
5
6
7
�
"metrics
trainable_variables
regularization_losses
	variables
#layer_regularization_losses

$layers
%non_trainable_variables
&layer_metrics
 
XV
VARIABLE_VALUEenc_0/kernel6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
enc_0/bias4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUE


0
1
 


0
1
�
'metrics
trainable_variables
regularization_losses
	variables
(layer_regularization_losses

)layers
*non_trainable_variables
+layer_metrics
XV
VARIABLE_VALUEenc_1/kernel6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
enc_1/bias4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
,metrics
trainable_variables
regularization_losses
	variables
-layer_regularization_losses

.layers
/non_trainable_variables
0layer_metrics
XV
VARIABLE_VALUEenc_2/kernel6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUE
TR
VARIABLE_VALUE
enc_2/bias4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
1metrics
trainable_variables
regularization_losses
	variables
2layer_regularization_losses

3layers
4non_trainable_variables
5layer_metrics
ZX
VARIABLE_VALUEenc_out/kernel6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUE
VT
VARIABLE_VALUEenc_out/bias4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUE

0
1
 

0
1
�
6metrics
trainable_variables
regularization_losses
 	variables
7layer_regularization_losses

8layers
9non_trainable_variables
:layer_metrics
 
 

0
1
2
3
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
 
z
serving_default_input_1Placeholder*'
_output_shapes
:���������*
dtype0*
shape:���������
�
StatefulPartitionedCallStatefulPartitionedCallserving_default_input_1enc_0/kernel
enc_0/biasenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_out/kernelenc_out/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� */
f*R(
&__inference_signature_wrapper_96902655
O
saver_filenamePlaceholder*
_output_shapes
: *
dtype0*
shape: 
�
StatefulPartitionedCall_1StatefulPartitionedCallsaver_filename enc_0/kernel/Read/ReadVariableOpenc_0/bias/Read/ReadVariableOp enc_1/kernel/Read/ReadVariableOpenc_1/bias/Read/ReadVariableOp enc_2/kernel/Read/ReadVariableOpenc_2/bias/Read/ReadVariableOp"enc_out/kernel/Read/ReadVariableOp enc_out/bias/Read/ReadVariableOpConst*
Tin
2
*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� **
f%R#
!__inference__traced_save_96903025
�
StatefulPartitionedCall_2StatefulPartitionedCallsaver_filenameenc_0/kernel
enc_0/biasenc_1/kernel
enc_1/biasenc_2/kernel
enc_2/biasenc_out/kernelenc_out/bias*
Tin
2	*
Tout
2*
_collective_manager_ids
 *
_output_shapes
: * 
_read_only_resource_inputs
 *-
config_proto

CPU

GPU 2J 8� *-
f(R&
$__inference__traced_restore_96903059��
�I
�
E__inference_encoder_layer_call_and_return_conditional_losses_96902765

inputs(
$enc_0_matmul_readvariableop_resource)
%enc_0_biasadd_readvariableop_resource(
$enc_1_matmul_readvariableop_resource)
%enc_1_biasadd_readvariableop_resource(
$enc_2_matmul_readvariableop_resource)
%enc_2_biasadd_readvariableop_resource*
&enc_out_matmul_readvariableop_resource+
'enc_out_biasadd_readvariableop_resource
identity��enc_0/BiasAdd/ReadVariableOp�enc_0/MatMul/ReadVariableOp�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�enc_1/BiasAdd/ReadVariableOp�enc_1/MatMul/ReadVariableOp�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�enc_2/BiasAdd/ReadVariableOp�enc_2/MatMul/ReadVariableOp�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�enc_out/BiasAdd/ReadVariableOp�enc_out/MatMul/ReadVariableOp�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/MatMul/ReadVariableOpReadVariableOp$enc_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
enc_0/MatMul/ReadVariableOp�
enc_0/MatMulMatMulinputs#enc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_0/MatMul�
enc_0/BiasAdd/ReadVariableOpReadVariableOp%enc_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
enc_0/BiasAdd/ReadVariableOp�
enc_0/BiasAddBiasAddenc_0/MatMul:product:0$enc_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_0/BiasAddg
	enc_0/EluEluenc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
	enc_0/Elu�
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
enc_1/MatMul/ReadVariableOp�
enc_1/MatMulMatMulenc_0/Elu:activations:0#enc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_1/MatMul�
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
enc_1/BiasAdd/ReadVariableOp�
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_1/BiasAddg
	enc_1/EluEluenc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
	enc_1/Elu�
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
enc_2/MatMul/ReadVariableOp�
enc_2/MatMulMatMulenc_1/Elu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_2/MatMul�
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
enc_2/BiasAdd/ReadVariableOp�
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_2/BiasAddg
	enc_2/EluEluenc_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
	enc_2/Elu�
enc_out/MatMul/ReadVariableOpReadVariableOp&enc_out_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
enc_out/MatMul/ReadVariableOp�
enc_out/MatMulMatMulenc_2/Elu:activations:0%enc_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
enc_out/MatMul�
enc_out/BiasAdd/ReadVariableOpReadVariableOp'enc_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
enc_out/BiasAdd/ReadVariableOp�
enc_out/BiasAddBiasAddenc_out/MatMul:product:0&enc_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
enc_out/BiasAdd�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$enc_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&enc_out_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentityenc_out/BiasAdd:output:0^enc_0/BiasAdd/ReadVariableOp^enc_0/MatMul/ReadVariableOp,^enc_0/kernel/Regularizer/Abs/ReadVariableOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp,^enc_1/kernel/Regularizer/Abs/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp,^enc_2/kernel/Regularizer/Abs/ReadVariableOp^enc_out/BiasAdd/ReadVariableOp^enc_out/MatMul/ReadVariableOp.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2<
enc_0/BiasAdd/ReadVariableOpenc_0/BiasAdd/ReadVariableOp2:
enc_0/MatMul/ReadVariableOpenc_0/MatMul/ReadVariableOp2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp2@
enc_out/BiasAdd/ReadVariableOpenc_out/BiasAdd/ReadVariableOp2>
enc_out/MatMul/ReadVariableOpenc_out/MatMul/ReadVariableOp2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�

*__inference_enc_out_layer_call_fn_96902934

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_enc_out_layer_call_and_return_conditional_losses_969023802
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
}
(__inference_enc_1_layer_call_fn_96902871

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_1_layer_call_and_return_conditional_losses_969023152
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�:
�
E__inference_encoder_layer_call_and_return_conditional_losses_96902469
input_1
enc_0_96902424
enc_0_96902426
enc_1_96902429
enc_1_96902431
enc_2_96902434
enc_2_96902436
enc_out_96902439
enc_out_96902441
identity��enc_0/StatefulPartitionedCall�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�enc_1/StatefulPartitionedCall�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�enc_2/StatefulPartitionedCall�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�enc_out/StatefulPartitionedCall�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/StatefulPartitionedCallStatefulPartitionedCallinput_1enc_0_96902424enc_0_96902426*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_0_layer_call_and_return_conditional_losses_969022822
enc_0/StatefulPartitionedCall�
enc_1/StatefulPartitionedCallStatefulPartitionedCall&enc_0/StatefulPartitionedCall:output:0enc_1_96902429enc_1_96902431*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_1_layer_call_and_return_conditional_losses_969023152
enc_1/StatefulPartitionedCall�
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_96902434enc_2_96902436*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_2_layer_call_and_return_conditional_losses_969023482
enc_2/StatefulPartitionedCall�
enc_out/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_out_96902439enc_out_96902441*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_enc_out_layer_call_and_return_conditional_losses_969023802!
enc_out/StatefulPartitionedCall�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_0_96902424*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_1_96902429*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_2_96902434*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_out_96902439*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentity(enc_out/StatefulPartitionedCall:output:0^enc_0/StatefulPartitionedCall,^enc_0/kernel/Regularizer/Abs/ReadVariableOp^enc_1/StatefulPartitionedCall,^enc_1/kernel/Regularizer/Abs/ReadVariableOp^enc_2/StatefulPartitionedCall,^enc_2/kernel/Regularizer/Abs/ReadVariableOp ^enc_out/StatefulPartitionedCall.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2>
enc_0/StatefulPartitionedCallenc_0/StatefulPartitionedCall2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp2B
enc_out/StatefulPartitionedCallenc_out/StatefulPartitionedCall2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
}
(__inference_enc_0_layer_call_fn_96902839

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_0_layer_call_and_return_conditional_losses_969022822
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_enc_1_layer_call_and_return_conditional_losses_96902315

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^enc_1/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
}
(__inference_enc_2_layer_call_fn_96902903

inputs
unknown
	unknown_0
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_2_layer_call_and_return_conditional_losses_969023482
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
!__inference__traced_save_96903025
file_prefix+
'savev2_enc_0_kernel_read_readvariableop)
%savev2_enc_0_bias_read_readvariableop+
'savev2_enc_1_kernel_read_readvariableop)
%savev2_enc_1_bias_read_readvariableop+
'savev2_enc_2_kernel_read_readvariableop)
%savev2_enc_2_bias_read_readvariableop-
)savev2_enc_out_kernel_read_readvariableop+
'savev2_enc_out_bias_read_readvariableop
savev2_const

identity_1��MergeV2Checkpoints�
StaticRegexFullMatchStaticRegexFullMatchfile_prefix"/device:CPU:**
_output_shapes
: *
pattern
^s3://.*2
StaticRegexFullMatchc
ConstConst"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B.part2
Constl
Const_1Const"/device:CPU:**
_output_shapes
: *
dtype0*
valueB B
_temp/part2	
Const_1�
SelectSelectStaticRegexFullMatch:output:0Const:output:0Const_1:output:0"/device:CPU:**
T0*
_output_shapes
: 2
Selectt

StringJoin
StringJoinfile_prefixSelect:output:0"/device:CPU:**
N*
_output_shapes
: 2

StringJoinZ

num_shardsConst*
_output_shapes
: *
dtype0*
value	B :2

num_shards
ShardedFilename/shardConst"/device:CPU:0*
_output_shapes
: *
dtype0*
value	B : 2
ShardedFilename/shard�
ShardedFilenameShardedFilenameStringJoin:output:0ShardedFilename/shard:output:0num_shards:output:0"/device:CPU:0*
_output_shapes
: 2
ShardedFilename�
SaveV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
SaveV2/tensor_names�
SaveV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
SaveV2/shape_and_slices�
SaveV2SaveV2ShardedFilename:filename:0SaveV2/tensor_names:output:0 SaveV2/shape_and_slices:output:0'savev2_enc_0_kernel_read_readvariableop%savev2_enc_0_bias_read_readvariableop'savev2_enc_1_kernel_read_readvariableop%savev2_enc_1_bias_read_readvariableop'savev2_enc_2_kernel_read_readvariableop%savev2_enc_2_bias_read_readvariableop)savev2_enc_out_kernel_read_readvariableop'savev2_enc_out_bias_read_readvariableopsavev2_const"/device:CPU:0*
_output_shapes
 *
dtypes
2	2
SaveV2�
&MergeV2Checkpoints/checkpoint_prefixesPackShardedFilename:filename:0^SaveV2"/device:CPU:0*
N*
T0*
_output_shapes
:2(
&MergeV2Checkpoints/checkpoint_prefixes�
MergeV2CheckpointsMergeV2Checkpoints/MergeV2Checkpoints/checkpoint_prefixes:output:0file_prefix"/device:CPU:0*
_output_shapes
 2
MergeV2Checkpointsr
IdentityIdentityfile_prefix^MergeV2Checkpoints"/device:CPU:0*
T0*
_output_shapes
: 2

Identitym

Identity_1IdentityIdentity:output:0^MergeV2Checkpoints*
T0*
_output_shapes
: 2

Identity_1"!

identity_1Identity_1:output:0*W
_input_shapesF
D: :P:P:PP:P:PP:P:P:: 2(
MergeV2CheckpointsMergeV2Checkpoints:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix:$ 

_output_shapes

:P: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:PP: 

_output_shapes
:P:$ 

_output_shapes

:P: 

_output_shapes
::	

_output_shapes
: 
�
�
*__inference_encoder_layer_call_fn_96902539
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_encoder_layer_call_and_return_conditional_losses_969025202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
__inference_loss_fn_2_969029678
4enc_2_kernel_regularizer_abs_readvariableop_resource
identity��+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp4enc_2_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
IdentityIdentity enc_2/kernel/Regularizer/mul:z:0,^enc_2/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp
�
�
*__inference_encoder_layer_call_fn_96902807

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_encoder_layer_call_and_return_conditional_losses_969025892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�%
�
$__inference__traced_restore_96903059
file_prefix!
assignvariableop_enc_0_kernel!
assignvariableop_1_enc_0_bias#
assignvariableop_2_enc_1_kernel!
assignvariableop_3_enc_1_bias#
assignvariableop_4_enc_2_kernel!
assignvariableop_5_enc_2_bias%
!assignvariableop_6_enc_out_kernel#
assignvariableop_7_enc_out_bias

identity_9��AssignVariableOp�AssignVariableOp_1�AssignVariableOp_2�AssignVariableOp_3�AssignVariableOp_4�AssignVariableOp_5�AssignVariableOp_6�AssignVariableOp_7�
RestoreV2/tensor_namesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*�
value�B�	B6layer_with_weights-0/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-0/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-1/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-1/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-2/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-2/bias/.ATTRIBUTES/VARIABLE_VALUEB6layer_with_weights-3/kernel/.ATTRIBUTES/VARIABLE_VALUEB4layer_with_weights-3/bias/.ATTRIBUTES/VARIABLE_VALUEB_CHECKPOINTABLE_OBJECT_GRAPH2
RestoreV2/tensor_names�
RestoreV2/shape_and_slicesConst"/device:CPU:0*
_output_shapes
:	*
dtype0*%
valueB	B B B B B B B B B 2
RestoreV2/shape_and_slices�
	RestoreV2	RestoreV2file_prefixRestoreV2/tensor_names:output:0#RestoreV2/shape_and_slices:output:0"/device:CPU:0*8
_output_shapes&
$:::::::::*
dtypes
2	2
	RestoreV2g
IdentityIdentityRestoreV2:tensors:0"/device:CPU:0*
T0*
_output_shapes
:2

Identity�
AssignVariableOpAssignVariableOpassignvariableop_enc_0_kernelIdentity:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOpk

Identity_1IdentityRestoreV2:tensors:1"/device:CPU:0*
T0*
_output_shapes
:2

Identity_1�
AssignVariableOp_1AssignVariableOpassignvariableop_1_enc_0_biasIdentity_1:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_1k

Identity_2IdentityRestoreV2:tensors:2"/device:CPU:0*
T0*
_output_shapes
:2

Identity_2�
AssignVariableOp_2AssignVariableOpassignvariableop_2_enc_1_kernelIdentity_2:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_2k

Identity_3IdentityRestoreV2:tensors:3"/device:CPU:0*
T0*
_output_shapes
:2

Identity_3�
AssignVariableOp_3AssignVariableOpassignvariableop_3_enc_1_biasIdentity_3:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_3k

Identity_4IdentityRestoreV2:tensors:4"/device:CPU:0*
T0*
_output_shapes
:2

Identity_4�
AssignVariableOp_4AssignVariableOpassignvariableop_4_enc_2_kernelIdentity_4:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_4k

Identity_5IdentityRestoreV2:tensors:5"/device:CPU:0*
T0*
_output_shapes
:2

Identity_5�
AssignVariableOp_5AssignVariableOpassignvariableop_5_enc_2_biasIdentity_5:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_5k

Identity_6IdentityRestoreV2:tensors:6"/device:CPU:0*
T0*
_output_shapes
:2

Identity_6�
AssignVariableOp_6AssignVariableOp!assignvariableop_6_enc_out_kernelIdentity_6:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_6k

Identity_7IdentityRestoreV2:tensors:7"/device:CPU:0*
T0*
_output_shapes
:2

Identity_7�
AssignVariableOp_7AssignVariableOpassignvariableop_7_enc_out_biasIdentity_7:output:0"/device:CPU:0*
_output_shapes
 *
dtype02
AssignVariableOp_79
NoOpNoOp"/device:CPU:0*
_output_shapes
 2
NoOp�

Identity_8Identityfile_prefix^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7^NoOp"/device:CPU:0*
T0*
_output_shapes
: 2

Identity_8�

Identity_9IdentityIdentity_8:output:0^AssignVariableOp^AssignVariableOp_1^AssignVariableOp_2^AssignVariableOp_3^AssignVariableOp_4^AssignVariableOp_5^AssignVariableOp_6^AssignVariableOp_7*
T0*
_output_shapes
: 2

Identity_9"!

identity_9Identity_9:output:0*5
_input_shapes$
": ::::::::2$
AssignVariableOpAssignVariableOp2(
AssignVariableOp_1AssignVariableOp_12(
AssignVariableOp_2AssignVariableOp_22(
AssignVariableOp_3AssignVariableOp_32(
AssignVariableOp_4AssignVariableOp_42(
AssignVariableOp_5AssignVariableOp_52(
AssignVariableOp_6AssignVariableOp_62(
AssignVariableOp_7AssignVariableOp_7:C ?

_output_shapes
: 
%
_user_specified_namefile_prefix
�:
�
E__inference_encoder_layer_call_and_return_conditional_losses_96902421
input_1
enc_0_96902293
enc_0_96902295
enc_1_96902326
enc_1_96902328
enc_2_96902359
enc_2_96902361
enc_out_96902391
enc_out_96902393
identity��enc_0/StatefulPartitionedCall�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�enc_1/StatefulPartitionedCall�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�enc_2/StatefulPartitionedCall�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�enc_out/StatefulPartitionedCall�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/StatefulPartitionedCallStatefulPartitionedCallinput_1enc_0_96902293enc_0_96902295*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_0_layer_call_and_return_conditional_losses_969022822
enc_0/StatefulPartitionedCall�
enc_1/StatefulPartitionedCallStatefulPartitionedCall&enc_0/StatefulPartitionedCall:output:0enc_1_96902326enc_1_96902328*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_1_layer_call_and_return_conditional_losses_969023152
enc_1/StatefulPartitionedCall�
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_96902359enc_2_96902361*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_2_layer_call_and_return_conditional_losses_969023482
enc_2/StatefulPartitionedCall�
enc_out/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_out_96902391enc_out_96902393*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_enc_out_layer_call_and_return_conditional_losses_969023802!
enc_out/StatefulPartitionedCall�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_0_96902293*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_1_96902326*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_2_96902359*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_out_96902391*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentity(enc_out/StatefulPartitionedCall:output:0^enc_0/StatefulPartitionedCall,^enc_0/kernel/Regularizer/Abs/ReadVariableOp^enc_1/StatefulPartitionedCall,^enc_1/kernel/Regularizer/Abs/ReadVariableOp^enc_2/StatefulPartitionedCall,^enc_2/kernel/Regularizer/Abs/ReadVariableOp ^enc_out/StatefulPartitionedCall.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2>
enc_0/StatefulPartitionedCallenc_0/StatefulPartitionedCall2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp2B
enc_out/StatefulPartitionedCallenc_out/StatefulPartitionedCall2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
&__inference_signature_wrapper_96902655
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *,
f'R%
#__inference__wrapped_model_969022612
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
E__inference_enc_out_layer_call_and_return_conditional_losses_96902380

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�*
�
#__inference__wrapped_model_96902261
input_10
,encoder_enc_0_matmul_readvariableop_resource1
-encoder_enc_0_biasadd_readvariableop_resource0
,encoder_enc_1_matmul_readvariableop_resource1
-encoder_enc_1_biasadd_readvariableop_resource0
,encoder_enc_2_matmul_readvariableop_resource1
-encoder_enc_2_biasadd_readvariableop_resource2
.encoder_enc_out_matmul_readvariableop_resource3
/encoder_enc_out_biasadd_readvariableop_resource
identity��$encoder/enc_0/BiasAdd/ReadVariableOp�#encoder/enc_0/MatMul/ReadVariableOp�$encoder/enc_1/BiasAdd/ReadVariableOp�#encoder/enc_1/MatMul/ReadVariableOp�$encoder/enc_2/BiasAdd/ReadVariableOp�#encoder/enc_2/MatMul/ReadVariableOp�&encoder/enc_out/BiasAdd/ReadVariableOp�%encoder/enc_out/MatMul/ReadVariableOp�
#encoder/enc_0/MatMul/ReadVariableOpReadVariableOp,encoder_enc_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02%
#encoder/enc_0/MatMul/ReadVariableOp�
encoder/enc_0/MatMulMatMulinput_1+encoder/enc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
encoder/enc_0/MatMul�
$encoder/enc_0/BiasAdd/ReadVariableOpReadVariableOp-encoder_enc_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder/enc_0/BiasAdd/ReadVariableOp�
encoder/enc_0/BiasAddBiasAddencoder/enc_0/MatMul:product:0,encoder/enc_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
encoder/enc_0/BiasAdd
encoder/enc_0/EluEluencoder/enc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
encoder/enc_0/Elu�
#encoder/enc_1/MatMul/ReadVariableOpReadVariableOp,encoder_enc_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02%
#encoder/enc_1/MatMul/ReadVariableOp�
encoder/enc_1/MatMulMatMulencoder/enc_0/Elu:activations:0+encoder/enc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
encoder/enc_1/MatMul�
$encoder/enc_1/BiasAdd/ReadVariableOpReadVariableOp-encoder_enc_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder/enc_1/BiasAdd/ReadVariableOp�
encoder/enc_1/BiasAddBiasAddencoder/enc_1/MatMul:product:0,encoder/enc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
encoder/enc_1/BiasAdd
encoder/enc_1/EluEluencoder/enc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
encoder/enc_1/Elu�
#encoder/enc_2/MatMul/ReadVariableOpReadVariableOp,encoder_enc_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02%
#encoder/enc_2/MatMul/ReadVariableOp�
encoder/enc_2/MatMulMatMulencoder/enc_1/Elu:activations:0+encoder/enc_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
encoder/enc_2/MatMul�
$encoder/enc_2/BiasAdd/ReadVariableOpReadVariableOp-encoder_enc_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02&
$encoder/enc_2/BiasAdd/ReadVariableOp�
encoder/enc_2/BiasAddBiasAddencoder/enc_2/MatMul:product:0,encoder/enc_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
encoder/enc_2/BiasAdd
encoder/enc_2/EluEluencoder/enc_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
encoder/enc_2/Elu�
%encoder/enc_out/MatMul/ReadVariableOpReadVariableOp.encoder_enc_out_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02'
%encoder/enc_out/MatMul/ReadVariableOp�
encoder/enc_out/MatMulMatMulencoder/enc_2/Elu:activations:0-encoder/enc_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/enc_out/MatMul�
&encoder/enc_out/BiasAdd/ReadVariableOpReadVariableOp/encoder_enc_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02(
&encoder/enc_out/BiasAdd/ReadVariableOp�
encoder/enc_out/BiasAddBiasAdd encoder/enc_out/MatMul:product:0.encoder/enc_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
encoder/enc_out/BiasAdd�
IdentityIdentity encoder/enc_out/BiasAdd:output:0%^encoder/enc_0/BiasAdd/ReadVariableOp$^encoder/enc_0/MatMul/ReadVariableOp%^encoder/enc_1/BiasAdd/ReadVariableOp$^encoder/enc_1/MatMul/ReadVariableOp%^encoder/enc_2/BiasAdd/ReadVariableOp$^encoder/enc_2/MatMul/ReadVariableOp'^encoder/enc_out/BiasAdd/ReadVariableOp&^encoder/enc_out/MatMul/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2L
$encoder/enc_0/BiasAdd/ReadVariableOp$encoder/enc_0/BiasAdd/ReadVariableOp2J
#encoder/enc_0/MatMul/ReadVariableOp#encoder/enc_0/MatMul/ReadVariableOp2L
$encoder/enc_1/BiasAdd/ReadVariableOp$encoder/enc_1/BiasAdd/ReadVariableOp2J
#encoder/enc_1/MatMul/ReadVariableOp#encoder/enc_1/MatMul/ReadVariableOp2L
$encoder/enc_2/BiasAdd/ReadVariableOp$encoder/enc_2/BiasAdd/ReadVariableOp2J
#encoder/enc_2/MatMul/ReadVariableOp#encoder/enc_2/MatMul/ReadVariableOp2P
&encoder/enc_out/BiasAdd/ReadVariableOp&encoder/enc_out/BiasAdd/ReadVariableOp2N
%encoder/enc_out/MatMul/ReadVariableOp%encoder/enc_out/MatMul/ReadVariableOp:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�:
�
E__inference_encoder_layer_call_and_return_conditional_losses_96902589

inputs
enc_0_96902544
enc_0_96902546
enc_1_96902549
enc_1_96902551
enc_2_96902554
enc_2_96902556
enc_out_96902559
enc_out_96902561
identity��enc_0/StatefulPartitionedCall�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�enc_1/StatefulPartitionedCall�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�enc_2/StatefulPartitionedCall�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�enc_out/StatefulPartitionedCall�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_0_96902544enc_0_96902546*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_0_layer_call_and_return_conditional_losses_969022822
enc_0/StatefulPartitionedCall�
enc_1/StatefulPartitionedCallStatefulPartitionedCall&enc_0/StatefulPartitionedCall:output:0enc_1_96902549enc_1_96902551*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_1_layer_call_and_return_conditional_losses_969023152
enc_1/StatefulPartitionedCall�
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_96902554enc_2_96902556*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_2_layer_call_and_return_conditional_losses_969023482
enc_2/StatefulPartitionedCall�
enc_out/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_out_96902559enc_out_96902561*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_enc_out_layer_call_and_return_conditional_losses_969023802!
enc_out/StatefulPartitionedCall�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_0_96902544*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_1_96902549*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_2_96902554*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_out_96902559*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentity(enc_out/StatefulPartitionedCall:output:0^enc_0/StatefulPartitionedCall,^enc_0/kernel/Regularizer/Abs/ReadVariableOp^enc_1/StatefulPartitionedCall,^enc_1/kernel/Regularizer/Abs/ReadVariableOp^enc_2/StatefulPartitionedCall,^enc_2/kernel/Regularizer/Abs/ReadVariableOp ^enc_out/StatefulPartitionedCall.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2>
enc_0/StatefulPartitionedCallenc_0/StatefulPartitionedCall2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp2B
enc_out/StatefulPartitionedCallenc_out/StatefulPartitionedCall2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_enc_1_layer_call_and_return_conditional_losses_96902862

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^enc_1/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
E__inference_enc_out_layer_call_and_return_conditional_losses_96902925

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2	
BiasAdd�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentityBiasAdd:output:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�:
�
E__inference_encoder_layer_call_and_return_conditional_losses_96902520

inputs
enc_0_96902475
enc_0_96902477
enc_1_96902480
enc_1_96902482
enc_2_96902485
enc_2_96902487
enc_out_96902490
enc_out_96902492
identity��enc_0/StatefulPartitionedCall�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�enc_1/StatefulPartitionedCall�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�enc_2/StatefulPartitionedCall�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�enc_out/StatefulPartitionedCall�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/StatefulPartitionedCallStatefulPartitionedCallinputsenc_0_96902475enc_0_96902477*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_0_layer_call_and_return_conditional_losses_969022822
enc_0/StatefulPartitionedCall�
enc_1/StatefulPartitionedCallStatefulPartitionedCall&enc_0/StatefulPartitionedCall:output:0enc_1_96902480enc_1_96902482*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_1_layer_call_and_return_conditional_losses_969023152
enc_1/StatefulPartitionedCall�
enc_2/StatefulPartitionedCallStatefulPartitionedCall&enc_1/StatefulPartitionedCall:output:0enc_2_96902485enc_2_96902487*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������P*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *L
fGRE
C__inference_enc_2_layer_call_and_return_conditional_losses_969023482
enc_2/StatefulPartitionedCall�
enc_out/StatefulPartitionedCallStatefulPartitionedCall&enc_2/StatefulPartitionedCall:output:0enc_out_96902490enc_out_96902492*
Tin
2*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������*$
_read_only_resource_inputs
*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_enc_out_layer_call_and_return_conditional_losses_969023802!
enc_out/StatefulPartitionedCall�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_0_96902475*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_1_96902480*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_2_96902485*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpenc_out_96902490*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentity(enc_out/StatefulPartitionedCall:output:0^enc_0/StatefulPartitionedCall,^enc_0/kernel/Regularizer/Abs/ReadVariableOp^enc_1/StatefulPartitionedCall,^enc_1/kernel/Regularizer/Abs/ReadVariableOp^enc_2/StatefulPartitionedCall,^enc_2/kernel/Regularizer/Abs/ReadVariableOp ^enc_out/StatefulPartitionedCall.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2>
enc_0/StatefulPartitionedCallenc_0/StatefulPartitionedCall2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp2>
enc_1/StatefulPartitionedCallenc_1/StatefulPartitionedCall2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp2>
enc_2/StatefulPartitionedCallenc_2/StatefulPartitionedCall2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp2B
enc_out/StatefulPartitionedCallenc_out/StatefulPartitionedCall2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�I
�
E__inference_encoder_layer_call_and_return_conditional_losses_96902710

inputs(
$enc_0_matmul_readvariableop_resource)
%enc_0_biasadd_readvariableop_resource(
$enc_1_matmul_readvariableop_resource)
%enc_1_biasadd_readvariableop_resource(
$enc_2_matmul_readvariableop_resource)
%enc_2_biasadd_readvariableop_resource*
&enc_out_matmul_readvariableop_resource+
'enc_out_biasadd_readvariableop_resource
identity��enc_0/BiasAdd/ReadVariableOp�enc_0/MatMul/ReadVariableOp�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�enc_1/BiasAdd/ReadVariableOp�enc_1/MatMul/ReadVariableOp�+enc_1/kernel/Regularizer/Abs/ReadVariableOp�enc_2/BiasAdd/ReadVariableOp�enc_2/MatMul/ReadVariableOp�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�enc_out/BiasAdd/ReadVariableOp�enc_out/MatMul/ReadVariableOp�-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/MatMul/ReadVariableOpReadVariableOp$enc_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
enc_0/MatMul/ReadVariableOp�
enc_0/MatMulMatMulinputs#enc_0/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_0/MatMul�
enc_0/BiasAdd/ReadVariableOpReadVariableOp%enc_0_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
enc_0/BiasAdd/ReadVariableOp�
enc_0/BiasAddBiasAddenc_0/MatMul:product:0$enc_0/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_0/BiasAddg
	enc_0/EluEluenc_0/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
	enc_0/Elu�
enc_1/MatMul/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
enc_1/MatMul/ReadVariableOp�
enc_1/MatMulMatMulenc_0/Elu:activations:0#enc_1/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_1/MatMul�
enc_1/BiasAdd/ReadVariableOpReadVariableOp%enc_1_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
enc_1/BiasAdd/ReadVariableOp�
enc_1/BiasAddBiasAddenc_1/MatMul:product:0$enc_1/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_1/BiasAddg
	enc_1/EluEluenc_1/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
	enc_1/Elu�
enc_2/MatMul/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
enc_2/MatMul/ReadVariableOp�
enc_2/MatMulMatMulenc_1/Elu:activations:0#enc_2/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_2/MatMul�
enc_2/BiasAdd/ReadVariableOpReadVariableOp%enc_2_biasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
enc_2/BiasAdd/ReadVariableOp�
enc_2/BiasAddBiasAddenc_2/MatMul:product:0$enc_2/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
enc_2/BiasAddg
	enc_2/EluEluenc_2/BiasAdd:output:0*
T0*'
_output_shapes
:���������P2
	enc_2/Elu�
enc_out/MatMul/ReadVariableOpReadVariableOp&enc_out_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02
enc_out/MatMul/ReadVariableOp�
enc_out/MatMulMatMulenc_2/Elu:activations:0%enc_out/MatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
enc_out/MatMul�
enc_out/BiasAdd/ReadVariableOpReadVariableOp'enc_out_biasadd_readvariableop_resource*
_output_shapes
:*
dtype02 
enc_out/BiasAdd/ReadVariableOp�
enc_out/BiasAddBiasAddenc_out/MatMul:product:0&enc_out/BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������2
enc_out/BiasAdd�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$enc_0_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$enc_1_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp$enc_2_matmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp&enc_out_matmul_readvariableop_resource*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentityenc_out/BiasAdd:output:0^enc_0/BiasAdd/ReadVariableOp^enc_0/MatMul/ReadVariableOp,^enc_0/kernel/Regularizer/Abs/ReadVariableOp^enc_1/BiasAdd/ReadVariableOp^enc_1/MatMul/ReadVariableOp,^enc_1/kernel/Regularizer/Abs/ReadVariableOp^enc_2/BiasAdd/ReadVariableOp^enc_2/MatMul/ReadVariableOp,^enc_2/kernel/Regularizer/Abs/ReadVariableOp^enc_out/BiasAdd/ReadVariableOp^enc_out/MatMul/ReadVariableOp.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::2<
enc_0/BiasAdd/ReadVariableOpenc_0/BiasAdd/ReadVariableOp2:
enc_0/MatMul/ReadVariableOpenc_0/MatMul/ReadVariableOp2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp2<
enc_1/BiasAdd/ReadVariableOpenc_1/BiasAdd/ReadVariableOp2:
enc_1/MatMul/ReadVariableOpenc_1/MatMul/ReadVariableOp2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp2<
enc_2/BiasAdd/ReadVariableOpenc_2/BiasAdd/ReadVariableOp2:
enc_2/MatMul/ReadVariableOpenc_2/MatMul/ReadVariableOp2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp2@
enc_out/BiasAdd/ReadVariableOpenc_out/BiasAdd/ReadVariableOp2>
enc_out/MatMul/ReadVariableOpenc_out/MatMul/ReadVariableOp2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_enc_0_layer_call_and_return_conditional_losses_96902282

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^enc_0/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
__inference_loss_fn_1_969029568
4enc_1_kernel_regularizer_abs_readvariableop_resource
identity��+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
+enc_1/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp4enc_1_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_1/kernel/Regularizer/Abs/ReadVariableOp�
enc_1/kernel/Regularizer/AbsAbs3enc_1/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_1/kernel/Regularizer/Abs�
enc_1/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_1/kernel/Regularizer/Const�
enc_1/kernel/Regularizer/SumSum enc_1/kernel/Regularizer/Abs:y:0'enc_1/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/Sum�
enc_1/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_1/kernel/Regularizer/mul/x�
enc_1/kernel/Regularizer/mulMul'enc_1/kernel/Regularizer/mul/x:output:0%enc_1/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_1/kernel/Regularizer/mul�
IdentityIdentity enc_1/kernel/Regularizer/mul:z:0,^enc_1/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Z
+enc_1/kernel/Regularizer/Abs/ReadVariableOp+enc_1/kernel/Regularizer/Abs/ReadVariableOp
�
�
__inference_loss_fn_0_969029458
4enc_0_kernel_regularizer_abs_readvariableop_resource
identity��+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp4enc_0_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
IdentityIdentity enc_0/kernel/Regularizer/mul:z:0,^enc_0/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp
�
�
*__inference_encoder_layer_call_fn_96902786

inputs
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinputsunknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_encoder_layer_call_and_return_conditional_losses_969025202
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
*__inference_encoder_layer_call_fn_96902608
input_1
unknown
	unknown_0
	unknown_1
	unknown_2
	unknown_3
	unknown_4
	unknown_5
	unknown_6
identity��StatefulPartitionedCall�
StatefulPartitionedCallStatefulPartitionedCallinput_1unknown	unknown_0	unknown_1	unknown_2	unknown_3	unknown_4	unknown_5	unknown_6*
Tin
2	*
Tout
2*
_collective_manager_ids
 *'
_output_shapes
:���������**
_read_only_resource_inputs

*-
config_proto

CPU

GPU 2J 8� *N
fIRG
E__inference_encoder_layer_call_and_return_conditional_losses_969025892
StatefulPartitionedCall�
IdentityIdentity StatefulPartitionedCall:output:0^StatefulPartitionedCall*
T0*'
_output_shapes
:���������2

Identity"
identityIdentity:output:0*F
_input_shapes5
3:���������::::::::22
StatefulPartitionedCallStatefulPartitionedCall:P L
'
_output_shapes
:���������
!
_user_specified_name	input_1
�
�
C__inference_enc_0_layer_call_and_return_conditional_losses_96902830

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
+enc_0/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:P*
dtype02-
+enc_0/kernel/Regularizer/Abs/ReadVariableOp�
enc_0/kernel/Regularizer/AbsAbs3enc_0/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2
enc_0/kernel/Regularizer/Abs�
enc_0/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_0/kernel/Regularizer/Const�
enc_0/kernel/Regularizer/SumSum enc_0/kernel/Regularizer/Abs:y:0'enc_0/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/Sum�
enc_0/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_0/kernel/Regularizer/mul/x�
enc_0/kernel/Regularizer/mulMul'enc_0/kernel/Regularizer/mul/x:output:0%enc_0/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_0/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^enc_0/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+enc_0/kernel/Regularizer/Abs/ReadVariableOp+enc_0/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������
 
_user_specified_nameinputs
�
�
C__inference_enc_2_layer_call_and_return_conditional_losses_96902348

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^enc_2/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
C__inference_enc_2_layer_call_and_return_conditional_losses_96902894

inputs"
matmul_readvariableop_resource#
biasadd_readvariableop_resource
identity��BiasAdd/ReadVariableOp�MatMul/ReadVariableOp�+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
MatMul/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02
MatMul/ReadVariableOps
MatMulMatMulinputsMatMul/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2
MatMul�
BiasAdd/ReadVariableOpReadVariableOpbiasadd_readvariableop_resource*
_output_shapes
:P*
dtype02
BiasAdd/ReadVariableOp�
BiasAddBiasAddMatMul:product:0BiasAdd/ReadVariableOp:value:0*
T0*'
_output_shapes
:���������P2	
BiasAddU
EluEluBiasAdd:output:0*
T0*'
_output_shapes
:���������P2
Elu�
+enc_2/kernel/Regularizer/Abs/ReadVariableOpReadVariableOpmatmul_readvariableop_resource*
_output_shapes

:PP*
dtype02-
+enc_2/kernel/Regularizer/Abs/ReadVariableOp�
enc_2/kernel/Regularizer/AbsAbs3enc_2/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:PP2
enc_2/kernel/Regularizer/Abs�
enc_2/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2 
enc_2/kernel/Regularizer/Const�
enc_2/kernel/Regularizer/SumSum enc_2/kernel/Regularizer/Abs:y:0'enc_2/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/Sum�
enc_2/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2 
enc_2/kernel/Regularizer/mul/x�
enc_2/kernel/Regularizer/mulMul'enc_2/kernel/Regularizer/mul/x:output:0%enc_2/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2
enc_2/kernel/Regularizer/mul�
IdentityIdentityElu:activations:0^BiasAdd/ReadVariableOp^MatMul/ReadVariableOp,^enc_2/kernel/Regularizer/Abs/ReadVariableOp*
T0*'
_output_shapes
:���������P2

Identity"
identityIdentity:output:0*.
_input_shapes
:���������P::20
BiasAdd/ReadVariableOpBiasAdd/ReadVariableOp2.
MatMul/ReadVariableOpMatMul/ReadVariableOp2Z
+enc_2/kernel/Regularizer/Abs/ReadVariableOp+enc_2/kernel/Regularizer/Abs/ReadVariableOp:O K
'
_output_shapes
:���������P
 
_user_specified_nameinputs
�
�
__inference_loss_fn_3_96902978:
6enc_out_kernel_regularizer_abs_readvariableop_resource
identity��-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
-enc_out/kernel/Regularizer/Abs/ReadVariableOpReadVariableOp6enc_out_kernel_regularizer_abs_readvariableop_resource*
_output_shapes

:P*
dtype02/
-enc_out/kernel/Regularizer/Abs/ReadVariableOp�
enc_out/kernel/Regularizer/AbsAbs5enc_out/kernel/Regularizer/Abs/ReadVariableOp:value:0*
T0*
_output_shapes

:P2 
enc_out/kernel/Regularizer/Abs�
 enc_out/kernel/Regularizer/ConstConst*
_output_shapes
:*
dtype0*
valueB"       2"
 enc_out/kernel/Regularizer/Const�
enc_out/kernel/Regularizer/SumSum"enc_out/kernel/Regularizer/Abs:y:0)enc_out/kernel/Regularizer/Const:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/Sum�
 enc_out/kernel/Regularizer/mul/xConst*
_output_shapes
: *
dtype0*
valueB
 *
�#<2"
 enc_out/kernel/Regularizer/mul/x�
enc_out/kernel/Regularizer/mulMul)enc_out/kernel/Regularizer/mul/x:output:0'enc_out/kernel/Regularizer/Sum:output:0*
T0*
_output_shapes
: 2 
enc_out/kernel/Regularizer/mul�
IdentityIdentity"enc_out/kernel/Regularizer/mul:z:0.^enc_out/kernel/Regularizer/Abs/ReadVariableOp*
T0*
_output_shapes
: 2

Identity"
identityIdentity:output:0*
_input_shapes
:2^
-enc_out/kernel/Regularizer/Abs/ReadVariableOp-enc_out/kernel/Regularizer/Abs/ReadVariableOp"�L
saver_filename:0StatefulPartitionedCall_1:0StatefulPartitionedCall_28"
saved_model_main_op

NoOp*>
__saved_model_init_op%#
__saved_model_init_op

NoOp*�
serving_default�
;
input_10
serving_default_input_1:0���������;
enc_out0
StatefulPartitionedCall:0���������tensorflow/serving/predict:��
�)
layer_with_weights-0
layer-0
layer_with_weights-1
layer-1
layer_with_weights-2
layer-2
layer_with_weights-3
layer-3
trainable_variables
regularization_losses
	variables
	keras_api
	
signatures
;_default_save_signature
<__call__
*=&call_and_return_all_conditional_losses"�&
_tf_keras_sequential�&{"class_name": "Sequential", "name": "encoder", "trainable": true, "expects_training_arg": true, "dtype": "float32", "batch_input_shape": null, "must_restore_from_config": false, "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "enc_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "enc_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "enc_2", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "enc_out", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}, "is_graph_network": true, "keras_version": "2.4.0", "backend": "tensorflow", "model_config": {"class_name": "Sequential", "config": {"name": "encoder", "layers": [{"class_name": "InputLayer", "config": {"batch_input_shape": {"class_name": "__tuple__", "items": [null, 3]}, "dtype": "float32", "sparse": false, "ragged": false, "name": "input_1"}}, {"class_name": "Dense", "config": {"name": "enc_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "enc_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "enc_2", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}, {"class_name": "Dense", "config": {"name": "enc_out", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}}]}}}
�


kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
>__call__
*?&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "enc_0", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_0", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 3}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 3]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
@__call__
*A&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "enc_1", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_1", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�

kernel
bias
trainable_variables
regularization_losses
	variables
	keras_api
B__call__
*C&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "enc_2", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_2", "trainable": true, "dtype": "float32", "units": 80, "activation": "elu", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
�

kernel
bias
trainable_variables
regularization_losses
 	variables
!	keras_api
D__call__
*E&call_and_return_all_conditional_losses"�
_tf_keras_layer�{"class_name": "Dense", "name": "enc_out", "trainable": true, "expects_training_arg": false, "dtype": "float32", "batch_input_shape": null, "stateful": false, "must_restore_from_config": false, "config": {"name": "enc_out", "trainable": true, "dtype": "float32", "units": 2, "activation": "linear", "use_bias": true, "kernel_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "bias_initializer": {"class_name": "HeUniform", "config": {"seed": null}}, "kernel_regularizer": {"class_name": "L1", "config": {"l1": 0.009999999776482582}}, "bias_regularizer": null, "activity_regularizer": null, "kernel_constraint": null, "bias_constraint": null}, "input_spec": {"class_name": "InputSpec", "config": {"dtype": null, "shape": null, "ndim": null, "max_ndim": null, "min_ndim": 2, "axes": {"-1": 80}}}, "build_input_shape": {"class_name": "TensorShape", "items": [null, 80]}}
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
<
F0
G1
H2
I3"
trackable_list_wrapper
X

0
1
2
3
4
5
6
7"
trackable_list_wrapper
�
"metrics
trainable_variables
regularization_losses
	variables
#layer_regularization_losses

$layers
%non_trainable_variables
&layer_metrics
<__call__
;_default_save_signature
*=&call_and_return_all_conditional_losses
&="call_and_return_conditional_losses"
_generic_user_object
,
Jserving_default"
signature_map
:P2enc_0/kernel
:P2
enc_0/bias
.

0
1"
trackable_list_wrapper
'
F0"
trackable_list_wrapper
.

0
1"
trackable_list_wrapper
�
'metrics
trainable_variables
regularization_losses
	variables
(layer_regularization_losses

)layers
*non_trainable_variables
+layer_metrics
>__call__
*?&call_and_return_all_conditional_losses
&?"call_and_return_conditional_losses"
_generic_user_object
:PP2enc_1/kernel
:P2
enc_1/bias
.
0
1"
trackable_list_wrapper
'
G0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
,metrics
trainable_variables
regularization_losses
	variables
-layer_regularization_losses

.layers
/non_trainable_variables
0layer_metrics
@__call__
*A&call_and_return_all_conditional_losses
&A"call_and_return_conditional_losses"
_generic_user_object
:PP2enc_2/kernel
:P2
enc_2/bias
.
0
1"
trackable_list_wrapper
'
H0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
1metrics
trainable_variables
regularization_losses
	variables
2layer_regularization_losses

3layers
4non_trainable_variables
5layer_metrics
B__call__
*C&call_and_return_all_conditional_losses
&C"call_and_return_conditional_losses"
_generic_user_object
 :P2enc_out/kernel
:2enc_out/bias
.
0
1"
trackable_list_wrapper
'
I0"
trackable_list_wrapper
.
0
1"
trackable_list_wrapper
�
6metrics
trainable_variables
regularization_losses
 	variables
7layer_regularization_losses

8layers
9non_trainable_variables
:layer_metrics
D__call__
*E&call_and_return_all_conditional_losses
&E"call_and_return_conditional_losses"
_generic_user_object
 "
trackable_list_wrapper
 "
trackable_list_wrapper
<
0
1
2
3"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
F0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
G0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
H0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
 "
trackable_list_wrapper
'
I0"
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_list_wrapper
 "
trackable_dict_wrapper
�2�
#__inference__wrapped_model_96902261�
���
FullArgSpec
args� 
varargsjargs
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *&�#
!�
input_1���������
�2�
*__inference_encoder_layer_call_fn_96902608
*__inference_encoder_layer_call_fn_96902786
*__inference_encoder_layer_call_fn_96902539
*__inference_encoder_layer_call_fn_96902807�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
E__inference_encoder_layer_call_and_return_conditional_losses_96902421
E__inference_encoder_layer_call_and_return_conditional_losses_96902765
E__inference_encoder_layer_call_and_return_conditional_losses_96902710
E__inference_encoder_layer_call_and_return_conditional_losses_96902469�
���
FullArgSpec1
args)�&
jself
jinputs

jtraining
jmask
varargs
 
varkw
 
defaults�
p 

 

kwonlyargs� 
kwonlydefaults� 
annotations� *
 
�2�
(__inference_enc_0_layer_call_fn_96902839�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_enc_0_layer_call_and_return_conditional_losses_96902830�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_enc_1_layer_call_fn_96902871�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_enc_1_layer_call_and_return_conditional_losses_96902862�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
(__inference_enc_2_layer_call_fn_96902903�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
C__inference_enc_2_layer_call_and_return_conditional_losses_96902894�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
*__inference_enc_out_layer_call_fn_96902934�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
E__inference_enc_out_layer_call_and_return_conditional_losses_96902925�
���
FullArgSpec
args�
jself
jinputs
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 
�2�
__inference_loss_fn_0_96902945�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_1_96902956�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_2_96902967�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�2�
__inference_loss_fn_3_96902978�
���
FullArgSpec
args� 
varargs
 
varkw
 
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *� 
�B�
&__inference_signature_wrapper_96902655input_1"�
���
FullArgSpec
args� 
varargs
 
varkwjkwargs
defaults
 

kwonlyargs� 
kwonlydefaults
 
annotations� *
 �
#__inference__wrapped_model_96902261o
0�-
&�#
!�
input_1���������
� "1�.
,
enc_out!�
enc_out����������
C__inference_enc_0_layer_call_and_return_conditional_losses_96902830\
/�,
%�"
 �
inputs���������
� "%�"
�
0���������P
� {
(__inference_enc_0_layer_call_fn_96902839O
/�,
%�"
 �
inputs���������
� "����������P�
C__inference_enc_1_layer_call_and_return_conditional_losses_96902862\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� {
(__inference_enc_1_layer_call_fn_96902871O/�,
%�"
 �
inputs���������P
� "����������P�
C__inference_enc_2_layer_call_and_return_conditional_losses_96902894\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������P
� {
(__inference_enc_2_layer_call_fn_96902903O/�,
%�"
 �
inputs���������P
� "����������P�
E__inference_enc_out_layer_call_and_return_conditional_losses_96902925\/�,
%�"
 �
inputs���������P
� "%�"
�
0���������
� }
*__inference_enc_out_layer_call_fn_96902934O/�,
%�"
 �
inputs���������P
� "�����������
E__inference_encoder_layer_call_and_return_conditional_losses_96902421k
8�5
.�+
!�
input_1���������
p

 
� "%�"
�
0���������
� �
E__inference_encoder_layer_call_and_return_conditional_losses_96902469k
8�5
.�+
!�
input_1���������
p 

 
� "%�"
�
0���������
� �
E__inference_encoder_layer_call_and_return_conditional_losses_96902710j
7�4
-�*
 �
inputs���������
p

 
� "%�"
�
0���������
� �
E__inference_encoder_layer_call_and_return_conditional_losses_96902765j
7�4
-�*
 �
inputs���������
p 

 
� "%�"
�
0���������
� �
*__inference_encoder_layer_call_fn_96902539^
8�5
.�+
!�
input_1���������
p

 
� "�����������
*__inference_encoder_layer_call_fn_96902608^
8�5
.�+
!�
input_1���������
p 

 
� "�����������
*__inference_encoder_layer_call_fn_96902786]
7�4
-�*
 �
inputs���������
p

 
� "�����������
*__inference_encoder_layer_call_fn_96902807]
7�4
-�*
 �
inputs���������
p 

 
� "����������=
__inference_loss_fn_0_96902945
�

� 
� "� =
__inference_loss_fn_1_96902956�

� 
� "� =
__inference_loss_fn_2_96902967�

� 
� "� =
__inference_loss_fn_3_96902978�

� 
� "� �
&__inference_signature_wrapper_96902655z
;�8
� 
1�.
,
input_1!�
input_1���������"1�.
,
enc_out!�
enc_out���������