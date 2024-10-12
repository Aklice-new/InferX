# 记录一下debug过程中的问题

1. 在Allocator类中，一开始直接用std::map来保存 内存的指针(void*)和内存区域大小的关系，然后在查找的过程中段错误。
   问题是因为std::map中的键值需要重载比较运算符，来维护RB-tree中的关系，void*不能直接来比较，所以出现了问题。改成了std::unordered_map。