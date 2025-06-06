// RUN: %clang_analyze_cc1 -analyzer-checker=alpha.webkit.UncountedLocalVarsChecker -verify %s

#include "mock-types.h"
#include "mock-system-header.h"

void someFunction();

namespace raw_ptr {
void foo() {
  RefCountable *bar;
  // FIXME: later on we might warn on uninitialized vars too
}

void bar(RefCountable *) {}
} // namespace raw_ptr

namespace reference {
void foo_ref() {
  RefCountable automatic;
  RefCountable &bar = automatic;
  // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  someFunction();
  bar.method();
}

void foo_ref_trivial() {
  RefCountable automatic;
  RefCountable &bar = automatic;
}

void bar_ref(RefCountable &) {}
} // namespace reference

namespace guardian_scopes {
void foo1() {
  RefPtr<RefCountable> foo;
  { RefCountable *bar = foo.get(); }
}

void foo2() {
  RefPtr<RefCountable> foo;
  // missing embedded scope here
  RefCountable *bar = foo.get();
  // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  someFunction();
  bar->method();
}

void foo3() {
  RefPtr<RefCountable> foo;
  {
    { RefCountable *bar = foo.get(); }
  }
}

void foo4() {
  {
    RefPtr<RefCountable> foo;
    { RefCountable *bar = foo.get(); }
  }
}

void foo5() {
  RefPtr<RefCountable> foo;
  auto* bar = foo.get();
  bar->trivial();
}

void foo6() {
  RefPtr<RefCountable> foo;
  auto* bar = foo.get();
  // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  bar->method();
}

struct SelfReferencingStruct {
  SelfReferencingStruct* ptr;
  RefCountable* obj { nullptr };
};

void foo7(RefCountable* obj) {
  SelfReferencingStruct bar = { &bar, obj };
  bar.obj->method();
}

void foo8(RefCountable* obj) {
  RefPtr<RefCountable> foo;
  {
    RefCountable *bar = foo.get();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    foo = nullptr;
    bar->method();
  }
  RefPtr<RefCountable> baz;
  {
    RefCountable *bar = baz.get();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    baz = obj;
    bar->method();
  }
  foo = nullptr;
  {
    RefCountable *bar = foo.get();
    // No warning. It's okay to mutate RefPtr in an outer scope.
    bar->method();
  }
  foo = obj;
  {
    RefCountable *bar = foo.get();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    foo.releaseNonNull();
    bar->method();
  }
  {
    RefCountable *bar = foo.get();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    foo = obj ? obj : nullptr;
    bar->method();
  }
  {
    RefCountable *bar = foo->trivial() ? foo.get() : nullptr;
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    foo = nullptr;
    bar->method();
  }
}

void foo9(RefCountable& o) {
  Ref<RefCountable> guardian(o);
  {
    RefCountable &bar = guardian.get();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    guardian = o; // We don't detect that we're setting it to the same value.
    bar.method();
  }
  {
    RefCountable *bar = guardian.ptr();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    Ref<RefCountable> other(*bar); // We don't detect other has the same value as guardian.
    guardian.swap(other);
    bar->method();
  }
  {
    RefCountable *bar = guardian.ptr();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    Ref<RefCountable> other(static_cast<Ref<RefCountable>&&>(guardian));
    bar->method();
  }
  {
    RefCountable *bar = guardian.ptr();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    guardian.leakRef();
    bar->method();
  }
  {
    RefCountable *bar = guardian.ptr();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    guardian = o.trivial() ? o : *bar;
    bar->method();
  }
}

} // namespace guardian_scopes

namespace auto_keyword {
class Foo {
  RefCountable *provide_ref_ctnbl();

  void evil_func() {
    RefCountable *bar = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    auto *baz = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'baz' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    auto *baz2 = this->provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'baz2' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    [[clang::suppress]] auto *baz_suppressed = provide_ref_ctnbl(); // no-warning
  }

  void func() {
    RefCountable *bar = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    if (bar)
      bar->method();
  }
};
} // namespace auto_keyword

namespace guardian_casts {
void foo1() {
  RefPtr<RefCountable> foo;
  {
    RefCountable *bar = downcast<RefCountable>(foo.get());
    bar->method();
  }
  foo->method();
}

void foo2() {
  RefPtr<RefCountable> foo;
  {
    RefCountable *bar =
        static_cast<RefCountable *>(downcast<RefCountable>(foo.get()));
    someFunction();
  }
}
} // namespace guardian_casts

namespace casts {

RefCountable* provide() { return nullptr; }
RefCountable* downcast(RefCountable*);
template<class T> T* bitwise_cast(T*);
template<class T> T* bit_cast(T*);

  void foo() {
    auto* cast1 = downcast(provide());
    auto* cast2 = bitwise_cast(provide());
    auto* cast3 = bit_cast(provide());
   }
} // namespace casts

namespace guardian_ref_conversion_operator {
void foo() {
  Ref<RefCountable> rc;
  {
    RefCountable &rr = rc;
    rr.method();
    someFunction();
  }
}
} // namespace guardian_ref_conversion_operator

namespace ignore_for_if {
RefCountable *provide_ref_ctnbl() { return nullptr; }

void foo() {
  // no warnings
  if (RefCountable *a = provide_ref_ctnbl())
    a->trivial();
  for (RefCountable *b = provide_ref_ctnbl(); b != nullptr;)
    b->trivial();
  RefCountable *array[1];
  for (RefCountable *c : array)
    c->trivial();
  while (RefCountable *d = provide_ref_ctnbl())
    d->trivial();
  do {
    RefCountable *e = provide_ref_ctnbl();
    e->trivial();
  } while (1);
  someFunction();
}

void bar() {
  if (RefCountable *a = provide_ref_ctnbl()) {
    // expected-warning@-1{{Local variable 'a' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    a->method();    
  }
  for (RefCountable *b = provide_ref_ctnbl(); b != nullptr;) {
    // expected-warning@-1{{Local variable 'b' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    b->method();
  }
  RefCountable *array[1];
  for (RefCountable *c : array) {
    // expected-warning@-1{{Local variable 'c' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    c->method();
  }

  while (RefCountable *d = provide_ref_ctnbl()) {
    // expected-warning@-1{{Local variable 'd' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    d->method();
  }
  do {
    RefCountable *e = provide_ref_ctnbl();
    // expected-warning@-1{{Local variable 'e' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    e->method();
  } while (1);
  someFunction();
}

} // namespace ignore_for_if

namespace ignore_system_headers {

RefCountable *provide_ref_ctnbl();

void system_header() {
  localVar<RefCountable>(provide_ref_ctnbl);
}

} // ignore_system_headers

namespace conditional_op {
RefCountable *provide_ref_ctnbl();
bool bar();

void foo() {
  RefCountable *a = bar() ? nullptr : provide_ref_ctnbl();
  // expected-warning@-1{{Local variable 'a' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  RefPtr<RefCountable> b = provide_ref_ctnbl();
  {
    RefCountable* c = bar() ? nullptr : b.get();
    c->method();
    RefCountable* d = bar() ? b.get() : nullptr;
    d->method();
  }
}

} // namespace conditional_op

namespace local_assignment_basic {

RefCountable *provide_ref_cntbl();

void foo(RefCountable* a) {
  RefCountable* b = a;
  // expected-warning@-1{{Local variable 'b' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  if (b->trivial())
    b = provide_ref_cntbl();
}

void bar(RefCountable* a) {
  RefCountable* b;
  // expected-warning@-1{{Local variable 'b' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  b = provide_ref_cntbl();
}

void baz() {
  RefPtr a = provide_ref_cntbl();
  {
    RefCountable* b = a.get();
    // expected-warning@-1{{Local variable 'b' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    b = provide_ref_cntbl();
  }
}

} // namespace local_assignment_basic

namespace local_assignment_to_parameter {

RefCountable *provide_ref_cntbl();
void someFunction();

void foo(RefCountable* a) {
  a = provide_ref_cntbl();
  // expected-warning@-1{{Assignment to an uncounted parameter 'a' is unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  someFunction();
  a->method();
}

} // namespace local_assignment_to_parameter

namespace local_assignment_to_static_local {

RefCountable *provide_ref_cntbl();
void someFunction();

void foo() {
  static RefCountable* a = nullptr;
  // expected-warning@-1{{Static local variable 'a' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  a = provide_ref_cntbl();
  someFunction();
  a->method();
}

} // namespace local_assignment_to_static_local

namespace local_assignment_to_global {

RefCountable *provide_ref_cntbl();
void someFunction();

RefCountable* g_a = nullptr;
// expected-warning@-1{{Global variable 'local_assignment_to_global::g_a' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}

void foo() {
  g_a = provide_ref_cntbl();
  someFunction();
  g_a->method();
}

} // namespace local_assignment_to_global

namespace local_refcountable_checkable_object {

RefCountableAndCheckable* provide_obj();

void local_raw_ptr() {
  RefCountableAndCheckable* a = nullptr;
  // expected-warning@-1{{Local variable 'a' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  a = provide_obj();
  a->method();
}

void local_checked_ptr() {
  CheckedPtr<RefCountableAndCheckable> a = nullptr;
  a = provide_obj();
  a->method();
}

void local_var_with_guardian_checked_ptr() {
  CheckedPtr<RefCountableAndCheckable> a = provide_obj();
  {
    auto* b = a.get();
    b->method();
  }
}

void local_var_with_guardian_checked_ptr_with_assignment() {
  CheckedPtr<RefCountableAndCheckable> a = provide_obj();
  {
    RefCountableAndCheckable* b = a.get();
    // expected-warning@-1{{Local variable 'b' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    b = provide_obj();
    b->method();
  }
}

void local_var_with_guardian_checked_ref() {
  CheckedRef<RefCountableAndCheckable> a = *provide_obj();
  {
    RefCountableAndCheckable& b = a;
    b.method();
  }
}

void static_var() {
  static RefCountableAndCheckable* a = nullptr;
  // expected-warning@-1{{Static local variable 'a' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  a = provide_obj();
}

} // namespace local_refcountable_checkable_object

namespace local_var_in_recursive_function {

struct TreeNode {
  Ref<TreeNode> create() { return Ref(*new TreeNode); }

  void ref() const { ++refCount; }
  void deref() const {
    if (!--refCount)
      delete this;
  }

  int recursiveCost();
  int recursiveWeight();
  int weight();

  int cost { 0 };
  mutable unsigned refCount { 0 };
  TreeNode* nextSibling { nullptr };
  TreeNode* firstChild { nullptr };
};

int TreeNode::recursiveCost() {
  // no warnings
  unsigned totalCost = cost;
  for (TreeNode* node = firstChild; node; node = node->nextSibling)
    totalCost += recursiveCost();
  return totalCost;
}

int TreeNode::recursiveWeight() {
  unsigned totalCost = weight();
  for (TreeNode* node = firstChild; node; node = node->nextSibling)
    // expected-warning@-1{{Local variable 'node' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    totalCost += recursiveWeight();
  return totalCost;
}

} // namespace local_var_in_recursive_function

namespace local_var_for_singleton {
  RefCountable *singleton();
  RefCountable *otherSingleton();
  void foo() {
    RefCountable* bar = singleton();
    RefCountable* baz = otherSingleton();
  }
}

namespace virtual_function {
  struct SomeObject {
    virtual RefCountable* provide() { return nullptr; }
    virtual RefCountable* operator&() { return nullptr; }
  };
  void foo(SomeObject* obj) {
    auto* bar = obj->provide();
    // expected-warning@-1{{Local variable 'bar' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
    auto* baz = &*obj;
    // expected-warning@-1{{Local variable 'baz' is uncounted and unsafe [alpha.webkit.UncountedLocalVarsChecker]}}
  }
}