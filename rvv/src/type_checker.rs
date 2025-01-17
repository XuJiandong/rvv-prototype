use std::collections::HashMap;

use anyhow::anyhow;

use crate::ast::{
    BareFnArg, Block, Expression, FnArg, ItemFn, Pattern, ReturnType, Signature, Span, Statement,
    Type, TypedExpression, WithSpan,
};
use crate::SpannedError;

#[derive(Debug)]
pub struct VarInfo {
    pub span: Span,
    pub mut_token: Option<Span>,
    pub ty: Option<Box<WithSpan<Type>>>,
    // indicate the varable liftime
    pub start_expr_id: usize,
    pub end_expr_id: usize,
}

impl VarInfo {
    pub fn new(
        span: Span,
        mut_token: Option<Span>,
        ty: Option<Box<WithSpan<Type>>>,
        start_expr_id: usize,
    ) -> VarInfo {
        VarInfo {
            span,
            mut_token,
            ty,
            start_expr_id,
            end_expr_id: start_expr_id,
        }
    }

    #[allow(dead_code)]
    pub fn is_unused(&self) -> bool {
        self.start_expr_id == self.end_expr_id
    }
}

#[derive(Default, Debug)]
pub struct CheckerContext {
    expr_id: usize,
    pub literal_exprs: HashMap<usize, syn::Lit>,
    pub uninfered_exprs: HashMap<usize, TypedExpression>,
    // ident => (mutability, Type)
    pub variables: HashMap<syn::Ident, VarInfo>,
}

impl CheckerContext {
    // zero expr_id is for function parameters
    fn next_expr_id(&mut self) -> usize {
        self.expr_id += 1;
        self.expr_id
    }
}

// ================================
// ==== impl TypeChecker for T ====
// ================================
// Extra Rules:
//   1. variable shadowing is forbidden
pub trait TypeChecker {
    // infer(by fill TypedExpression.{ty, id} field) and check types
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError>;
}

impl TypeChecker for ReturnType {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        match self {
            ReturnType::Default => {}
            ReturnType::Type(_span, ty) => {
                ty.0.check_types(context)?;
            }
        }
        Ok(())
    }
}

impl TypeChecker for BareFnArg {
    fn check_types(&mut self, _context: &mut CheckerContext) -> Result<(), SpannedError> {
        Ok(())
    }
}
impl TypeChecker for Type {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        match self {
            Type::Array { elem, len, .. } => {
                elem.check_types(context)?;
                len.check_types(context)?;
            }
            Type::BareFn { inputs, output, .. } => {
                for input in inputs {
                    input.check_types(context)?;
                }
                output.0.check_types(context)?;
            }
            Type::Path(_path) => {}
            Type::Reference { elem, .. } => {
                elem.0.check_types(context)?;
            }
            Type::Slice { elem, .. } => {
                elem.0.check_types(context)?;
            }
            Type::Tuple { elems, .. } => {
                for elem in elems {
                    elem.0.check_types(context)?;
                }
            }
        }
        Ok(())
    }
}
impl TypeChecker for Pattern {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        match self {
            Pattern::Ident { .. } => {}
            Pattern::Type { pat, ty, .. } => {
                pat.0.check_types(context)?;
                ty.0.check_types(context)?;
            }
            Pattern::Range { lo, hi, .. } => {
                lo.check_types(context)?;
                hi.check_types(context)?;
            }
            Pattern::Path(_path) => {}
            Pattern::Wild(_) => {}
        }
        Ok(())
    }
}
impl TypeChecker for Expression {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        match self {
            Expression::Array(_) => {}
            Expression::Assign { left, right, .. } => {
                left.check_types(context)?;
                right.check_types(context)?;
            }
            Expression::AssignOp { left, right, .. } => {
                left.check_types(context)?;
                right.check_types(context)?;
            }
            Expression::Binary { left, right, .. } => {
                left.check_types(context)?;
                right.check_types(context)?;
            }
            Expression::Call { func, args, .. } => {
                func.check_types(context)?;
                for arg in args {
                    arg.check_types(context)?;
                }
            }
            Expression::MethodCall { receiver, args, .. } => {
                receiver.check_types(context)?;
                for arg in args {
                    arg.check_types(context)?;
                }
            }
            Expression::Macro(_) => {}
            Expression::Unary { expr, .. } => {
                expr.check_types(context)?;
            }
            Expression::Field { base, .. } => {
                base.check_types(context)?;
            }
            Expression::Cast { expr, ty, .. } => {
                expr.check_types(context)?;
                ty.0.check_types(context)?;
            }
            Expression::Repeat { expr, len, .. } => {
                expr.check_types(context)?;
                len.check_types(context)?;
            }
            Expression::Lit(_) => {}
            Expression::Paren { expr, .. } => {
                expr.check_types(context)?;
            }
            Expression::Reference { expr, .. } => {
                expr.check_types(context)?;
            }
            Expression::Index { expr, index, .. } => {
                expr.check_types(context)?;
                index.check_types(context)?;
            }
            Expression::Path(_path) => {}
            Expression::Break(_) => {}
            Expression::Continue(_) => {}
            Expression::Return { expr, .. } => {
                if let Some(expr) = expr.as_mut() {
                    expr.check_types(context)?;
                }
            }
            Expression::Block(block) => {
                block.check_types(context)?;
            }
            Expression::If {
                cond,
                then_branch,
                else_branch,
                ..
            } => {
                cond.check_types(context)?;
                then_branch.check_types(context)?;
                if let Some((_span, expr)) = else_branch.as_mut() {
                    expr.check_types(context)?;
                }
            }
            Expression::Range { from, to, .. } => {
                if let Some(expr) = from.as_mut() {
                    expr.check_types(context)?;
                }
                if let Some(expr) = to.as_mut() {
                    expr.check_types(context)?;
                }
            }
            Expression::Loop { body, .. } => {
                body.check_types(context)?;
            }
            Expression::ForLoop {
                pat, expr, body, ..
            } => {
                pat.0.check_types(context)?;
                expr.check_types(context)?;
                body.check_types(context)?;
            }
        }
        Ok(())
    }
}
impl TypeChecker for TypedExpression {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        if self.id != usize::max_value() {
            panic!(
                "[Bug]: TypedExpression.id assigned more than once: id={}, expr={:?}",
                self.id, self.expr
            );
        }
        self.expr.0.check_types(context)?;
        self.id = context.next_expr_id();
        self.ty = match &mut self.expr.0 {
            Expression::Assign { left, right, .. } | Expression::AssignOp { left, right, .. } => {
                match (&mut left.ty, &mut right.ty) {
                    (Some(left_ty), Some(right_ty)) => {
                        if left_ty.0 != right_ty.0 {
                            return Err((self.expr.1, anyhow!("Assign/AssignOp with different types is not supported in rvv_vector. left={}, right={}",
                                                             left_ty.0.type_name().unwrap_or_else(|| "unknown".to_string()),
                                                             right_ty.0.type_name().unwrap_or_else(|| "unknown".to_string())
                            )));
                        }
                    }
                    (None, Some(_)) => {
                        left.ty = right.ty.clone();
                    }
                    (Some(_), None) => {
                        right.ty = left.ty.clone();
                    }
                    (None, None) => {}
                }
                Some(Box::new((Type::unit(), Span::default())))
            }
            Expression::Binary { left, op, right } => {
                let bool_op = matches!(
                    op,
                    syn::BinOp::Eq(_)
                        | syn::BinOp::Lt(_)
                        | syn::BinOp::Le(_)
                        | syn::BinOp::Ne(_)
                        | syn::BinOp::Ge(_)
                        | syn::BinOp::Gt(_)
                );

                let inner_ty = match (&mut left.ty, &mut right.ty) {
                    (Some(left_ty), Some(right_ty)) => {
                        if left_ty.0 != right_ty.0 {
                            return Err((
                                self.expr.1,
                                anyhow!(
                                    "Binary op with different types is not supported in rvv_vector. left={}, right={}",
                                    left_ty.0.type_name().unwrap_or_else(|| "unknown".to_string()),
                                    right_ty.0.type_name().unwrap_or_else(|| "unknown".to_string())
                                ),
                            ));
                        }
                        left.ty.clone()
                    }
                    (None, Some(_)) => {
                        left.ty = right.ty.clone();
                        right.ty.clone()
                    }
                    (Some(_), None) => {
                        right.ty = left.ty.clone();
                        left.ty.clone()
                    }
                    (None, None) => None,
                };

                if bool_op {
                    Some(Box::new((Type::primitive("bool"), Span::default())))
                } else {
                    inner_ty
                }
            }
            Expression::MethodCall {
                receiver, method, ..
            } => match receiver.type_name().as_deref() {
                Some("U256") | Some("U512") | Some("U1024") => match method.to_string().as_str() {
                    "wrapping_add" | "wrapping_sub" | "wrapping_mul" | "wrapping_div"
                    | "wrapping_rem" | "saturating_add" | "saturating_sub" | "saturating_mul" => {
                        receiver.ty.clone()
                    }
                    "checked_add" | "checked_sub" | "checked_mul" | "checked_div"
                    | "checked_rem" => None,
                    "overflowing_add" | "overflowing_sub" | "overflowing_mul" => Some(Box::new((
                        Type::Tuple {
                            paren_token: Span::default(),
                            elems: vec![
                                receiver
                                    .ty
                                    .as_ref()
                                    .map(|ty| WithSpan::<Type>::clone(ty))
                                    .unwrap(),
                                (Type::primitive("bool"), Span::default()),
                            ],
                        },
                        Span::default(),
                    ))),
                    _ => None,
                },
                _ => None,
            },
            // Handle:
            //   - x.overflowing_mul(y).0
            //   - x.overflowing_mul(y).1
            // NOTE: Just infer the type, not used in codegen.
            Expression::Field {
                base,
                member: syn::Member::Unnamed(syn::Index { index, .. }),
                ..
            } => {
                let index_usize = *index as usize;
                match base.ty.as_ref().map(|span_ty| &span_ty.0) {
                    Some(Type::Tuple { elems, .. }) if index_usize < elems.len() => {
                        Some(Box::new(elems[index_usize].clone()))
                    }
                    _ => None,
                }
            }
            Expression::Unary { op, expr } => match op {
                syn::UnOp::Deref(_) => {
                    if !expr.ty.as_ref().map(|ty| ty.0.is_ref()).unwrap_or(true) {
                        return Err((expr.expr.1, anyhow!("deref a variable that is not reference is not supported in rvv_vector")));
                    }
                    expr.ty
                        .as_ref()
                        .and_then(|ty| ty.0.clone().into_deref())
                        .map(|(_mutability, ty)| ty)
                }
                syn::UnOp::Not(_) | syn::UnOp::Neg(_) => None,
            },
            Expression::Paren { expr, .. } => expr.ty.clone(),
            Expression::Reference {
                and_token,
                mutability,
                expr,
            } => {
                if expr.ty.as_ref().map(|ty| ty.0.is_ref()).unwrap_or(false) {
                    return Err((
                        self.expr.1,
                        anyhow!("multiple reference to a variable is not supported in rvv_vector"),
                    ));
                }
                expr.ty.clone().map(|ty| {
                    Box::new((
                        ty.0.into_ref(*and_token, None, *mutability, ty.1),
                        Span::default(),
                    ))
                })
            }
            Expression::Block(block) => block.get_type(),
            Expression::If {
                then_branch,
                else_branch,
                ..
            } => {
                if let Some((_else_span, else_expr)) = else_branch.as_mut() {
                    let then_type = then_branch.get_type();
                    match (&then_type, &else_expr.ty) {
                        (Some(left_ty), Some(right_ty)) => {
                            if left_ty.0 != right_ty.0 {
                                return Err((
                                    self.expr.1,
                                    anyhow!("different if else branch types is not supported in rvv_vector. then-branch={}, else-branch={}",
                                            left_ty.0.type_name().unwrap_or_else(|| "unknown".to_string()),
                                            right_ty.0.type_name().unwrap_or_else(|| "unknown".to_string())
                                    ),
                                ));
                            }
                            Some(left_ty.clone())
                        }
                        (None, Some(_)) => else_expr.ty.clone(),
                        (Some(then_ty), None) => {
                            let ty = Some(then_ty.clone());
                            else_expr.ty = ty.clone();
                            ty
                        }
                        (None, None) => None,
                    }
                } else {
                    Some(Box::new((Type::unit(), Span::default())))
                }
            }
            Expression::Cast { ty, .. } => Some(ty.clone()),
            Expression::Path(path) => match path.get_ident() {
                Some(ident) => {
                    let current_expr_id = context.expr_id;
                    context.variables.get_mut(ident).and_then(|info| {
                        info.end_expr_id = current_expr_id;
                        info.ty.clone()
                    })
                }
                None => None,
            },
            Expression::Break(_) => Some(Box::new((Type::unit(), Span::default()))),
            Expression::Continue(_) => Some(Box::new((Type::unit(), Span::default()))),
            Expression::Loop { .. } => Some(Box::new((Type::unit(), Span::default()))),
            Expression::ForLoop { .. } => Some(Box::new((Type::unit(), Span::default()))),
            _ => None,
        };

        if let Some(lit) = self.expr.0.get_literal() {
            context.literal_exprs.insert(self.id, lit.clone());
        } else if self.ty.is_none() {
            context.uninfered_exprs.insert(self.id, self.clone());
        }
        Ok(())
    }
}
impl TypeChecker for Statement {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        match self {
            Statement::Local { pat, init, .. } => {
                pat.0.check_types(context)?;
                init.check_types(context)?;
                match &pat.0 {
                    Pattern::Ident { mutability, ident } => {
                        if context.variables.contains_key(ident) {
                            return Err((
                                ident.span().into(),
                                anyhow!(
                                    "variable ({}) shadowing is not supported in rvv_vector",
                                    ident
                                ),
                            ));
                        }
                        context.variables.insert(
                            (*ident).clone(),
                            VarInfo::new(pat.1, *mutability, init.ty.clone(), context.expr_id),
                        );
                    }
                    Pattern::Type { pat, ty, .. } => {
                        if let Pattern::Ident { mutability, ident } = &pat.0 {
                            if init.ty.is_none() {
                                init.ty = Some(ty.clone());
                            }
                            if context
                                .variables
                                .insert(
                                    ident.clone(),
                                    VarInfo::new(
                                        pat.1,
                                        *mutability,
                                        Some(ty.clone()),
                                        context.expr_id,
                                    ),
                                )
                                .is_some()
                            {
                                return Err((
                                    ident.span().into(),
                                    anyhow!(
                                        "variable ({}) shadowing is not supported in rvv_vector",
                                        ident
                                    ),
                                ));
                            }
                        }
                    }
                    _ => {}
                }
            }
            Statement::Expr(expr) => {
                expr.check_types(context)?;
            }
            Statement::Semi(expr) => {
                expr.check_types(context)?;
            }
        }
        Ok(())
    }
}
impl TypeChecker for Block {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        for (stmt, _span) in self.stmts.iter_mut() {
            stmt.check_types(context)?;
        }
        Ok(())
    }
}
impl TypeChecker for FnArg {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        self.ty.0.check_types(context)?;
        Ok(())
    }
}
impl TypeChecker for Signature {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        for input in self.inputs.iter_mut() {
            // let binding in signature
            context.variables.insert(
                input.name.clone(),
                VarInfo::new(
                    input.span,
                    input.mutability,
                    Some(input.ty.clone()),
                    context.expr_id,
                ),
            );
            input.check_types(context)?;
        }
        self.output.0.check_types(context)?;
        Ok(())
    }
}

impl TypeChecker for ItemFn {
    fn check_types(&mut self, context: &mut CheckerContext) -> Result<(), SpannedError> {
        self.sig.check_types(context)?;
        self.block.check_types(context)?;
        Ok(())
    }
}
