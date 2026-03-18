# Section 5: Related Work

## 5.1 Speculative Decoding

Speculative decoding was independently proposed by Leviathan et al. [1] and Chen et al. [2], establishing the draft-then-verify paradigm for lossless acceleration of autoregressive generation. The key insight is that verifying $K$ draft tokens in parallel costs approximately the same as generating one token, yielding up to $(K+1)\times$ speedup when acceptance rates are high.

**Draft model designs** have diversified significantly:
- **Independent models**: SpecInfer [3] and DistillSpec [4] train a separate smaller model for drafting, optimizing for speed-quality tradeoff.
- **Self-speculative methods**: Draft & Verify [5] and Medusa [6] use the target model's own earlier layers or lightweight prediction heads to generate drafts without a separate model.
- **Feature-reusing methods**: EAGLE [7] and EAGLE-2 [8] build draft models that reuse the target model's last-hidden-state features, achieving higher acceptance rates. EAGLE-3 [9] extends this to MoE models with a dedicated speculator.
- **Tree-structured speculation**: SpecInfer [3] and Sequoia [10] organize drafts as trees rather than chains, exploring multiple continuation paths simultaneously.

**Adaptive speculation depth**: Several works dynamically adjust $K$ based on runtime statistics. EAGLE-2 [8] uses a confidence-based stopping criterion. SpecTr [11] frames the draft length as a bandit problem. DistillSpec [4] uses the draft model's confidence scores.

All these methods assume the verification cost is approximately constant regardless of $K$ — an assumption that breaks for MoE models under CPU offloading, as we demonstrate. SpecMoE is **orthogonal** to draft model design and can be composed with any of these methods.

## 5.2 MoE Inference Optimization

**Expert parallelism and placement**: GShard [12] and Switch Transformers [13] pioneer expert parallelism across devices, where each device holds a subset of experts. DeepSpeed-MoE [14] optimizes inter-device communication with hierarchical all-to-all. Tutel [15] dynamically switches between all-to-all and all-gather based on expert utilization. These focus on multi-GPU distributed settings, whereas SpecMoE targets single-GPU with CPU offloading.

**Expert offloading**: Mixtral-Offloading [16] employs LRU caching and speculative expert loading for Mixtral-8×7B. Pre-gated MoE [17] predicts expert selections one layer ahead to overlap computation with expert loading. MoE-Infinity [18] uses activation-aware prefetching and expert-level caching for memory-constrained serving. These methods optimize single-token expert loading but do not address the multi-token amplification caused by speculative decoding.

**Expert pruning and merging**: MoE-Pruner [19] reduces the number of experts at the model level. Sparse Upcycling [20] and SMEAR [21] merge expert weights to reduce the active expert pool. These are static optimizations that trade quality for efficiency, whereas SpecMoE preserves model quality by optimizing the serving system.

**Kernel-level optimization**: Megablocks [22] introduces block-sparse expert execution for batched MoE inference. The vLLM project's fused_moe kernel [23] uses Triton to fuse expert dispatch, FFN computation, and output scattering. SpecMoE's SpecFusedMoE extends these kernels with cross-token deduplication specific to the verify batch.

## 5.3 MoE × Speculative Decoding

The intersection of MoE and SD is an emerging area with limited prior work:

**MoE-Spec** [24] observes the expert activation amplification problem and proposes budgeting the total expert activation count across the verify batch. However, it enforces the budget by truncating expert selections (selecting fewer than $k$ experts for some tokens), which degrades output quality.

**Cascade Speculative Drafting** [25] dynamically adjusts the speculation depth $K$ based on estimated utility, which indirectly mitigates the MAF penalty by reducing $K$ when the cost is high. However, it cannot reduce the per-expert loading cost and may oscillate as it searches for the optimal $K$.

**SP-MoE** [26] separates prefill and decode scheduling for MoE models to reduce interference, but does not address the expert amplification within a single verify batch.

**MoE-SpAc** [27] proposes speculative execution of expert loading based on predicted routing, overlapping PCIe transfers with computation. This is similar in spirit to SpecMoE's expert prefetch but differs in two key ways: (1) MoE-SpAc does not exploit cross-token deduplication, and (2) it does not perform layer-level early termination.

SpecMoE addresses all three dimensions simultaneously — deduplication, early termination, and prefetching — at the operator level, providing a comprehensive solution that subsumes the benefits of prior scheduling-level approaches.

## 5.4 Operator-Level Inference Co-design

**FlashAttention** [28] pioneered the operator-co-design approach, fusing attention score computation and softmax into a single kernel that avoids materializing the full attention matrix. This demonstrated that operator-level optimization can deliver order-of-magnitude improvements that are invisible to scheduling-level approaches.

**FlashDecoding** [29] extends this to the decoding phase with split-KV parallelism. **PagedAttention** [30] (vLLM) redesigns KV cache management at the page granularity. These works share SpecMoE's philosophy of optimizing the operator internals while maintaining a transparent interface to the serving system.

SpecMoE applies this co-design principle to the MoE operator in the speculative decoding context — a combination that, to our knowledge, has not been explored in prior work. Our SpecFusedMoE kernel demonstrates that cross-token awareness within the MoE operator can eliminate redundant expert transfers, analogous to how FlashAttention eliminated redundant attention matrix materialization.

---

## References

[1] Y. Leviathan, M. Kalman, and Y. Matias, "Fast inference from transformers via speculative decoding," in ICML, 2023.

[2] C. Chen, S. Borgeaud, G. Irving, J.-B. Lespiau, L. Sifre, and J. Jumper, "Accelerating large language model decoding with speculative sampling," arXiv:2302.01318, 2023.

[3] X. Miao, G. Oliaro, Z. Zhang, X. Cheng, Z. Wang, R. Y. Y. Wong, Z. Zhu, D. Shi, G. Campbell, and Z. Jia, "SpecInfer: Accelerating generative large language model serving with tree-based speculative inference and verification," in ASPLOS, 2024.

[4] Y. Zhou, K. Lyu, A. S. Rawat, A. K. Menon, A. Rostamizadeh, S. Kumar, J.-F. Kagy, and A. Agarwal, "DistillSpec: Improving speculative decoding via knowledge distillation," in ICLR, 2024.

[5] J. Zhang, J. Wang, H. Li, L. Shou, K. Chen, G. Chen, and S. Mehrotra, "Draft & Verify: Lossless large language model acceleration via self-speculative decoding," in ACL, 2024.

[6] T. Cai, Y. Li, Z. Geng, H. Peng, J. D. Lee, D. Chen, and T. Dao, "Medusa: Simple LLM inference acceleration framework with multiple decoding heads," in ICML, 2024.

[7] Y. Li, F. Wei, C. Zhang, and H. Zhang, "EAGLE: Speculative sampling requires rethinking feature uncertainty," in ICML, 2024.

[8] Y. Li, F. Wei, C. Zhang, and H. Zhang, "EAGLE-2: Faster inference of language models with dynamic draft trees," in EMNLP, 2024.

[9] Y. Li, Y. Zhong, et al., "EAGLE-3: Scaling up inference acceleration of large language models via training-free speculative sampling," 2025.

[10] Z. Chen, X. Miao, Z. Zhang, and Z. Jia, "Sequoia: Scalable, robust, and hardware-aware speculative decoding," in NeurIPS, 2024.

[11] M. Sun, X. Zhong, H. Jia, and Z. Dong, "SpecTr: Fast speculative decoding via optimal transport," in NeurIPS, 2023.

[12] D. Lepikhin, H. Lee, Y. Xu, D. Chen, O. Firat, Y. Huang, M. Krikun, N. Shazeer, and Z. Chen, "GShard: Scaling giant models with conditional computation and automatic sharding," in ICLR, 2021.

[13] W. Fedus, B. Zoph, and N. Shazeer, "Switch transformers: Scaling to trillion parameter models with simple and efficient sparsity," JMLR, 2022.

[14] S. Rajbhandari, C. Li, Z. Yao, M. Zhang, R. Y. Aminabadi, A. A. Awan, J. Rasley, and Y. He, "DeepSpeed-MoE: Advancing mixture-of-experts inference and training to power next-generation AI scale," in ICML, 2022.

[15] C. Hwang, W. Cui, Y. Xiong, Z. Yang, Z. Liu, H. Hu, Z. Wang, R. Salas, J. Jose, P. Ram, J. Chau, P. Cheng, F. Yang, M. Yang, and Y. Xiong, "Tutel: Adaptive mixture-of-experts at scale," in MLSys, 2023.

[16] E. Eliseev and D. Mazur, "Fast inference of mixture-of-experts language models with offloading," arXiv:2312.17238, 2023.

[17] S.-Y. Hwang, K. Yi, G. Tarolli, and N. Kim, "Pre-gated MoE: An algorithm-system co-design for fast and scalable mixture-of-experts inference," arXiv:2308.12066, 2023.

[18] L. Xue, J. Zheng, Y. Li, J. Liu, R. Chen, and S. Chen, "MoE-Infinity: Activation-aware expert offloading for efficient MoE serving," arXiv:2401.14361, 2024.

[19] R. Lu, et al., "Not all experts are equal: Efficient expert pruning and skipping for mixture of experts large language models," in ACL, 2024.

[20] A. Komatsuzaki, J. Puigcerver, J. Lee-Thorp, C. R. Ruiz, B. Mustafa, J. Ainslie, Y. Tay, M. Dehghani, and N. Houlsby, "Sparse upcycling: Training mixture-of-experts from dense checkpoints," in ICLR, 2023.

[21] Y. Muqeeth, H. Liu, Y. Liu, and C. Raffel, "Soft merging of experts with adaptive routing," TMLR, 2024.

[22] T. Gale, D. Narayanan, C. Young, and M. Zaharia, "MegaBlocks: Efficient sparse training with mixture-of-experts," in MLSys, 2023.

[23] W. Kwon, Z. Li, S. Zhuang, Y. Sheng, L. Zheng, C. H. Yu, J. E. Gonzalez, H. Zhang, and I. Stoica, "Efficient memory management for large language model serving with PagedAttention," in SOSP, 2023.

[24] S. Yi, et al., "MoE-Spec: MoE-based speculative decoding with expert-aware budgeting," 2025.

[25] Z. Chen, R. May, et al., "Cascade speculative drafting for even faster LLM inference," 2024.

[26] Y. Jin, et al., "SP-MoE: Efficient serving of mixture-of-experts models with separated prefill and decode," 2024.

[27] K. Park, et al., "MoE-SpAc: Speculative expert activation for efficient MoE serving," 2025.

[28] T. Dao, D. Y. Fu, S. Ermon, A. Rudra, and C. Ré, "FlashAttention: Fast and memory-efficient exact attention with IO-awareness," in NeurIPS, 2022.

[29] T. Dao, "FlashAttention-2: Faster attention with better parallelism and work partitioning," in ICLR, 2024.

[30] W. Kwon et al., "Efficient memory management for large language model serving with PagedAttention," in SOSP, 2023.
