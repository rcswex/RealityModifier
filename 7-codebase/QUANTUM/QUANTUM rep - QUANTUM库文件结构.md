# QUANTUM Repository Structure

```
quantum-asm/
├── README.md                          # Main documentation
├── LICENSE                            # MIT License
├── PATENTS                            # Patent notices
├── CONTRIBUTING.md                    # Contribution guidelines
├── CHANGELOG.md                       # Release history
├── CITATION.cff                       # Citation information
├── .github/                           # GitHub configuration
│   ├── workflows/
│   │   ├── ci.yml                   # Continuous integration
│   │   ├── release.yml              # Release automation
│   │   └── benchmarks.yml           # Performance benchmarks
│   ├── ISSUE_TEMPLATE/
│   │   └── bug_report.md            # Bug report template
│   └── PULL_REQUEST_TEMPLATE.md
│
├── src/                               # C++ source code
│   ├── core/                        # 🔴 CRITICAL: Core Quantum Implementation
│   │   ├── stable_quantum_gates.h   # 🔴 KEY FILE 1: Stable gate definitions
│   │   ├── stable_quantum_gates.cpp # 🔴 KEY FILE 1: Mathematical implementation
│   │   ├── spacetime_compression.h  # 🔴 KEY FILE 2: 4D→2D compression theory
│   │   ├── spacetime_compression.cpp# 🔴 KEY FILE 2: Compression implementation
│   │   ├── quantum_processor.h      # Main processor interface
│   │   ├── quantum_processor.cpp    # Processor implementation
│   │   ├── instruction_set.h        # Instruction definitions
│   │   └── instruction_set.cpp      # Instruction implementations
│   │
│   ├── bridge/                      # Quantum-Classical Bridge (核心桥接层)
│   │   ├── qbin_format.h            # QBIN binary format definitions
│   │   ├── qbin_format.cpp          # QBIN format implementation
│   │   ├── binary_emitter.h         # Binary code generation
│   │   ├── binary_emitter.cpp       # Stable op → binary instruction
│   │   ├── classical_executor.h     # Classical execution engine
│   │   ├── classical_executor.cpp   # Execute quantum ops on CPU
│   │   ├── stability_encoder.h      # Encode stability into binary
│   │   └── stability_encoder.cpp    # Damping/normalization in binary
│   │
│   ├── compiler/                    # Quantum → Binary Compiler
│   │   ├── quantum_compiler.h       # Main compiler interface
│   │   ├── quantum_compiler.cpp     # QASM → QBIN compilation
│   │   ├── optimization/
│   │   │   ├── stable_patterns.h    # Pre-compiled stable sequences
│   │   │   ├── stable_patterns.cpp  # Pattern recognition & optimization
│   │   │   ├── gate_fusion.cpp      # Fuse gates for efficiency
│   │   │   └── binary_optimizer.cpp # Optimize binary output
│   │   ├── parser/
│   │   │   ├── qasm_parser.cpp      # Parse quantum assembly
│   │   │   ├── qasm_lexer.cpp       # Lexical analysis
│   │   │   └── syntax_validator.cpp # Validate QASM syntax
│   │   └── codegen/
│   │       ├── hex_encoder.cpp      # Encode to hex format
│   │       ├── instruction_map.cpp  # Map quantum → binary ops
│   │       └── symbol_table.cpp     # Symbol resolution
│   │
│   ├── assembly/                    # (Moved non-critical assembly files)
│   │   ├── composite_ops.h          # Composite stable operations
│   │   ├── composite_ops.cpp        # Bell pairs, QFT, etc.
│   │   └── legacy/                  # Legacy implementations
│   │       ├── stable_gates.h       # (Old location)
│   │       └── stable_gates.cpp     # (Old location)
│   │
│   ├── execution/
│   │   ├── instruction_executor.h   # Instruction execution engine
│   │   ├── instruction_executor.cpp # Execute with natural damping
│   │   ├── register_manager.h       # Quantum register management
│   │   ├── register_manager.cpp     # 4D register allocation
│   │   ├── convergence_engine.h     # Convergence mechanisms
│   │   └── convergence_engine.cpp   # Natural stability enforcement
│   │
│   ├── binary/                      # Binary Execution Layer
│   │   ├── qbin_loader.h            # Load QBIN files
│   │   ├── qbin_loader.cpp          # Parse and validate QBIN
│   │   ├── binary_runtime.h         # Binary runtime system
│   │   ├── binary_runtime.cpp       # Execute compiled quantum programs
│   │   ├── complex_ops_simd.h       # SIMD-optimized complex ops
│   │   ├── complex_ops_simd.cpp     # AVX2/AVX512 implementations
│   │   ├── cache_optimizer.h        # Cache-friendly execution
│   │   └── cache_optimizer.cpp      # Optimize memory access patterns
│   │
│   ├── math/
│   │   ├── complex_ops.h            # Complex arithmetic
│   │   ├── complex_ops.cpp          # Norm-preserving operations
│   │   ├── hypersphere.h            # Hypersphere projections
│   │   ├── hypersphere.cpp          # Stable manifold mapping
│   │   ├── clifford_algebra.h       # Clifford operations
│   │   └── clifford_algebra.cpp     # Algebraic stability
│   │
│   └── utils/
│       ├── profiler.h               # Performance profiling
│       ├── profiler.cpp             # Execution metrics
│       ├── constants.h              # Physical/mathematical constants
│       ├── hex_utils.h              # Hexadecimal utilities
│       └── logger.cpp               # Logging utilities
│
├── include/                          # Public headers
│   └── quantum/
│       ├── quantum.h                # Main public API
│       ├── processor.h              # Processor interface
│       ├── instructions.h           # Instruction set
│       ├── types.h                  # Type definitions
│       ├── binary_bridge.h          # Binary bridge interface
│       ├── qbin.h                   # QBIN format interface
│       └── version.h                # Version information
│
├── python/                           # Python bindings
│   ├── quantum/
│   │   ├── __init__.py
│   │   ├── processor.py             # Processor wrapper
│   │   ├── instructions.py          # Instruction builders
│   │   ├── assembly.py              # Assembly parser
│   │   ├── compiler.py              # Python → QBIN compiler
│   │   ├── binary_executor.py      # Execute QBIN files
│   │   ├── patterns/                # Pre-built stable patterns
│   │   │   ├── __init__.py
│   │   │   ├── bell_states.py      # Bell state patterns
│   │   │   ├── qft.py               # Stable QFT
│   │   │   ├── grover.py            # Grover with convergence
│   │   │   └── vqe.py               # Stable VQE
│   │   └── visualization.py         # State visualization
│   ├── setup.py                     # Package configuration
│   └── requirements.txt             # Python dependencies
│
├── assembly/                         # Assembly language
│   ├── grammar/
│   │   ├── quantum.g4               # ANTLR grammar
│   │   └── lexer.g4                 # Lexical rules
│   ├── stdlib/                      # Standard library
│   │   ├── stable_gates.qasm        # Basic stable gates
│   │   ├── algorithms.qasm          # Common algorithms
│   │   └── patterns.qasm            # Reusable patterns
│   └── examples/
│       ├── hello_quantum.qasm       # Basic examples
│       ├── bell_pair.qasm           # EPR pair creation
│       ├── grover_search.qasm       # Database search
│       ├── shor_factoring.qasm      # Integer factorization
│       └── qft_8bit.qasm            # 8-qubit QFT
│
├── qbin/                             # 🔴 CRITICAL: QBIN format tools
│   ├── examples/                    # 🔴 KEY FILE 3: Binary examples
│   │   ├── bell_pair.qbin          # 🔴 KEY EXAMPLE: Compiled Bell pair
│   │   ├── bell_pair.qbin.hex      # Human-readable hex dump
│   │   ├── grover_4bit.qbin        # Compiled Grover search
│   │   ├── qft_stable.qbin         # Compiled QFT
│   │   └── README.md                # Explains the binary format
│   ├── specification/
│   │   ├── qbin_spec_v2.md         # QBIN format specification
│   │   ├── encoding_rules.md        # Encoding conventions
│   │   └── hex_format.md            # Hexadecimal format details
│   ├── tools/
│   │   ├── qbin_assembler.cpp      # QASM → QBIN assembler
│   │   ├── qbin_disassembler.cpp   # QBIN → QASM disassembler
│   │   ├── qbin_validator.cpp      # Validate QBIN files
│   │   ├── qbin_optimizer.cpp      # Optimize QBIN code
│   │   └── hex_viewer.cpp          # View QBIN in hex format
│
├── tests/                            # Test suite
│   ├── unit/
│   │   ├── test_complex_ops.cpp    # Complex number tests
│   │   ├── test_gates.cpp           # Gate stability tests
│   │   ├── test_convergence.cpp     # Convergence verification
│   │   ├── test_spacetime.cpp       # 4D encoding tests
│   │   ├── test_compression.cpp     # Complex compression tests
│   │   ├── test_binary_bridge.cpp   # Binary bridge tests
│   │   └── test_instructions.py     # Python API tests
│   ├── integration/
│   │   ├── test_algorithms.cpp      # Algorithm tests
│   │   ├── test_stability.cpp       # Stability guarantees
│   │   ├── test_compiler.cpp        # Compiler tests
│   │   ├── test_qbin_execution.cpp  # QBIN execution tests
│   │   └── test_performance.cpp     # Performance benchmarks
│   ├── assembly/
│   │   ├── test_parser.cpp          # Assembly parser tests
│   │   └── test_programs/           # Test assembly programs
│   └── binary/
│       ├── test_qbin_format.cpp     # QBIN format tests
│       ├── test_hex_encoding.cpp    # Hex encoding tests
│       └── test_classical_exec.cpp  # Classical execution tests
│
├── examples/                         # Example code
│   ├── critical_demos/              # 🔴 CRITICAL: Core demonstrations
│   │   ├── 01_first_quantum_gate.cpp    # Show first ever classical quantum gate
│   │   ├── 02_spacetime_compression.cpp # Demonstrate 4D→2D compression
│   │   ├── 03_execute_qbin.cpp          # Execute compiled QBIN file
│   │   └── README.md                    # Explains the breakthrough
│   ├── cpp/
│   │   ├── basic_quantum.cpp       # C++ basics
│   │   ├── custom_algorithm.cpp    # Custom stable algorithm
│   │   ├── binary_execution.cpp    # Execute QBIN files
│   │   └── performance_demo.cpp    # Performance demonstration
│   ├── python/
│   │   ├── getting_started.py      # Python quickstart
│   │   ├── quantum_ml.py           # Machine learning
│   │   ├── chemistry.py            # Molecular simulation
│   │   ├── compile_to_binary.py    # Compile to QBIN
│   │   └── optimization.py         # QAOA example
│   └── assembly/
│       ├── tutorials/               # Assembly tutorials
│       └── algorithms/              # Algorithm implementations
│
├── docs/                             # Documentation
│   ├── CRITICAL_FILES.md            # 🔴 Explains the 3 key files
│   ├── source/
│   │   ├── index.rst               # Documentation home
│   │   ├── quickstart.rst          # Getting started
│   │   ├── core_concepts/          # 🔴 Core concepts documentation
│   │   │   ├── stable_gates.rst    # How stable gates work
│   │   │   ├── spacetime_compression.rst # 4D→2D theory
│   │   │   └── qbin_format.rst     # Binary format details
│   │   ├── architecture/
│   │   │   ├── overview.rst        # Architecture overview
│   │   │   ├── stability.rst       # Stability mechanisms
│   │   │   ├── complex_encoding.rst # Complex geometry
│   │   │   ├── spacetime.rst       # 4D encoding
│   │   │   └── binary_bridge.rst   # Quantum-classical bridge
│   │   ├── qbin/
│   │   │   ├── format_spec.rst     # QBIN format specification
│   │   │   ├── compiler.rst        # Compilation process
│   │   │   └── execution.rst       # Binary execution
│   │   ├── api/
│   │   │   ├── cpp.rst             # C++ API reference
│   │   │   ├── python.rst          # Python API reference
│   │   │   ├── assembly.rst        # Assembly language
│   │   │   └── binary.rst          # Binary format API
│   │   └── theory/
│   │       ├── mathematical_foundation.rst
│   │       ├── convergence_proofs.rst
│   │       ├── compression_theory.rst
│   │       └── benchmarks.rst
│   ├── Makefile                     # Documentation build
│   └── requirements.txt             # Documentation dependencies
│
├── benchmarks/                       # Performance benchmarks
│   ├── algorithms/
│   │   ├── grover_bench.cpp        # Grover search benchmarks
│   │   ├── shor_bench.cpp          # Shor factoring benchmarks
│   │   ├── vqe_bench.cpp           # VQE benchmarks
│   │   └── qft_bench.cpp           # QFT benchmarks
│   ├── stability/
│   │   ├── convergence_test.cpp    # Convergence rate tests
│   │   └── error_accumulation.cpp  # Error analysis
│   ├── binary/
│   │   ├── compilation_speed.cpp   # Compilation benchmarks
│   │   ├── execution_speed.cpp     # Binary execution speed
│   │   └── compression_ratio.cpp   # Compression efficiency
│   └── run_benchmarks.py            # Benchmark runner
│
├── tools/                            # Development tools
│   ├── instruction_designer/        # Design new stable instructions
│   │   ├── designer.py             # Interactive designer
│   │   └── templates/              # Instruction templates
│   ├── stability_analyzer/          # Analyze stability properties
│   │   ├── analyzer.cpp            # Stability analysis
│   │   └── visualizer.py           # Visualization tools
│   ├── qbin_tools/                 # QBIN manipulation tools
│   │   ├── qbin_editor.py          # Edit QBIN files
│   │   ├── qbin_debugger.cpp       # Debug QBIN execution
│   │   └── pattern_extractor.py    # Extract common patterns
│   └── format.sh                    # Code formatting script
│
├── cmake/                            # CMake configuration
│   ├── FindComplex.cmake           # Complex number libraries
│   ├── QuantumConfig.cmake         # QUANTUM configuration
│   ├── QBINConfig.cmake            # QBIN support
│   └── CompilerOptions.cmake       # Compiler settings
│
├── scripts/                          # Utility scripts
│   ├── install.sh                  # Installation script
│   ├── generate_instructions.py    # Generate instruction variants
│   ├── compile_to_qbin.py          # Batch compile to QBIN
│   └── package_release.sh          # Release packaging
│
├── docker/                           # Docker configurations
│   ├── Dockerfile                  # Base image
│   ├── Dockerfile.dev              # Development environment
│   ├── Dockerfile.qbin             # QBIN tools image
│   └── docker-compose.yml          # Service composition
│
├── CMakeLists.txt                    # CMake main configuration
├── .gitignore                        # Git ignore rules
├── .clang-format                     # C++ code style
├── pyproject.toml                    # Python project config
└── VERSION                           # Version number
```

## Key Design Features

### 1. **Integrated Binary Bridge** (`src/bridge/`)

The quantum-classical bridge is a core component, not an afterthought. It provides:

- Complex number compression for efficient classical representation
- Stable binary encoding that preserves quantum properties
- Direct CPU execution of quantum operations

### 2. **QBIN Format** (`qbin/`)

A dedicated binary format for quantum programs:

- Hexadecimal encoding for compact representation
- Pre-compiled stable operation sequences
- Built-in stability guarantees at the binary level

### 3. **Compilation Pipeline** (`src/compiler/`)

Complete toolchain from quantum assembly to classical binary:

- Pattern recognition for common quantum algorithms
- Optimization passes that preserve stability
- Direct generation of CPU-executable code

### 4. **Binary Execution** (`src/binary/`)

High-performance classical execution engine:

- SIMD-optimized complex number operations
- Cache-friendly memory layouts
- Pre-computed stable operation tables

### 5. **Stability Throughout**

Every layer maintains stability guarantees:

- Assembly instructions include damping
- Binary format encodes stability parameters
- Execution preserves convergence properties

This structure ensures that QUANTUM is not just a theoretical framework but a practical system that can execute quantum algorithms efficiently on classical hardware while maintaining mathematical stability.