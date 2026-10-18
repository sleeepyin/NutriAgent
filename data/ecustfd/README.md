# ECUSTFD

原图、标注和密度表还没有。已有的对照 CSV 在 `experiments/diagnosis/d4_ecustfd_scale/reference_results.csv`。

重跑 `reference_scale.py` 时，把下面的内容放在本目录：

```
density.xls
Annotations/*.xml
```

脚本把新的对照表写回实验目录里的那份 CSV。
