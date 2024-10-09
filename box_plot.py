from matplotlib.font_manager import findSystemFonts

# 列出所有系统字体的路径
font_paths = findSystemFonts()
font_names = [fp.split('/')[-1] for fp in font_paths]
print(len(font_names))
# 搜索特定字体
search_font = "Juice ITC"
found_fonts = [fn for fn in font_names if search_font.lower() in fn.lower()]

print("Found Fonts:", found_fonts)
