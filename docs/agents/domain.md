# 領域文件入口

目前的模型範圍與 local／global 語意，以[整合規格](../specs/neighbor-integration.md)、[整合說明](../NEIGHBOR_INTEGRATION.md)及[驗證說明](../NEIGHBOR_VALIDATION.md)為入口。單步訓練探針的輸入與精度比較約定見[探針文件](../validation/train-step-probe.md)；生成軌跡的 Precision parity 約定見[比較器文件](../validation/precision-parity.md)。

領域術語（空間分區與 Training region）收錄於根目錄 [CONTEXT.md](../../CONTEXT.md)；不可輕易回退的決策記在 [docs/adr/](../adr/)，目前有 [0001 Training region 預設](../adr/0001-training-region-defaults.md)。新術語或決策直接補進這兩處；既有 GitHub issue 的 spec 仍依 [issue tracker](issue-tracker.md) 管理。
