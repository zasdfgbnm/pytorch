#pragma once

#include <torch/csrc/WindowsTorchApiMacro.h>

#include <torch/csrc/jit/fuser/common/ir.h>
#include <torch/csrc/jit/fuser/common/type.h>

/*
 * Creating an Expr node returns the node that was created. This is useful
 * if you already have things connected and are directly modifying Exprs.
 * However, creating Expr nodes directly requires you to instantiate all
 * intermediate values. Arith is intended to cover all Exprs, however,
 * its goal is to return a value that is the output of the Expr.
 */ 

namespace torch{
namespace jit{
namespace fuser{

//Return new value of type that v1 and v2 promotes to
TORCH_API Val* promote_new(Val* v1, Val* v2);

TORCH_API Val* cast_op(DataType dtype, Val* v1);
TORCH_API Val* unary_op(UnaryOpType type, Val* v1);
TORCH_API Val* binary_op(BinaryOpType type, Val* v1, Val* v2);

TORCH_API Val* add(Val* v1, Val* v2);
TORCH_API Val* sub(Val* v1, Val* v2);
TORCH_API Val* mul(Val* v1, Val* v2);
TORCH_API Val* div(Val* v1, Val* v2);
TORCH_API Val* mod(Val* v1, Val* v2);
TORCH_API Val* lt(Val* v1, Val* v2);
TORCH_API Val* ceilDiv(Val* v1, Val* v2);

}}}
