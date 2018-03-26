import dlc_bci as bci

train_̇input, train_̇target = bci.load(root='./data', store_local=True)

print(str(type(train_̇input)), train_̇input.size())
print(str(type(train_̇target)), train_̇target.size())
test_̇input, test_̇target = bci.load(root='./data', train = False)
print(str(type(test_̇input)), test_̇input.size())
print(str(type(test_̇target)), test_̇target.size())