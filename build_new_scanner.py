"""
Build a new Scanner.py with the complete condition builder from Add_Alert.py
"""

# Read the current Scanner.py
with open('pages/Scanner.py', 'r', encoding='utf-8') as f:
    scanner_lines = f.readlines()

# Read Add_Alert.py
with open('pages/Add_Alert.py', 'r', encoding='utf-8') as f:
    alert_lines = f.readlines()

# Build the new Scanner.py
new_scanner = []

# 1. Header: Everything up to the condition builder (lines 0-211)
new_scanner.extend(scanner_lines[:212])

# 2. Add the title for condition builder
new_scanner.append("# Condition Builder Interface\n")
new_scanner.append("st.markdown(\"### Build Conditions Using Dropdown Menus\")\n")
new_scanner.append("\n")

# 3. Add the complete condition builder from Add_Alert (lines 1661-2568)
# This includes all indicators with proper col1, col2, col3 structure
new_scanner.extend(alert_lines[1659:2570])  # Start from col1, col2, col3 = st.columns...

# 4. Replace the col2 section (text input) with scanner-specific logic
# Remove lines about text input and add button
new_scanner.append("\n")
new_scanner.append("    with col2:\n")
new_scanner.append("        st.markdown(\"<br>\", unsafe_allow_html=True)  # Add spacing\n")
new_scanner.append("        if st.button(\"➕ Add Condition\", key=\"add_condition\"):\n")
new_scanner.append("            if indicator and indicator.strip():\n")
new_scanner.append("                st.session_state.scanner_conditions.append(indicator.strip())\n")
new_scanner.append("                st.success(f\"✅ Added: {indicator.strip()}\")\n")
new_scanner.append("                st.rerun()\n")
new_scanner.append("\n")
new_scanner.append("    with col3:\n")
new_scanner.append("        st.markdown(\"<br>\", unsafe_allow_html=True)  # Add spacing\n")
new_scanner.append("        if st.button(\"🗑️ Clear All\", key=\"clear_all\"):\n")
new_scanner.append("            st.session_state.scanner_conditions = []\n")
new_scanner.append("            st.rerun()\n")
new_scanner.append("\n")

# 5. Add current conditions display
new_scanner.append("# Display current conditions\n")
new_scanner.append("st.divider()\n")
new_scanner.append("st.markdown(\"### 📋 Current Scan Conditions\")\n")
new_scanner.append("\n")
new_scanner.append("if st.session_state.scanner_conditions:\n")
new_scanner.append("    for i, cond in enumerate(st.session_state.scanner_conditions):\n")
new_scanner.append("        col1, col2 = st.columns([5, 1])\n")
new_scanner.append("        with col1:\n")
new_scanner.append("            st.code(f\"{i+1}. {cond}\", language=\"python\")\n")
new_scanner.append("        with col2:\n")
new_scanner.append("            if st.button(\"✖\", key=f\"remove_{i}\", help=\"Remove this condition\"):\n")
new_scanner.append("                st.session_state.scanner_conditions.pop(i)\n")
new_scanner.append("                st.rerun()\n")
new_scanner.append("\n")
new_scanner.append("    # Combination logic for multiple conditions\n")
new_scanner.append("    if len(st.session_state.scanner_conditions) > 1:\n")
new_scanner.append("        st.session_state.scanner_logic = st.text_input(\n")
new_scanner.append("            \"Combination Logic:\",\n")
new_scanner.append("            value=st.session_state.scanner_logic,\n")
new_scanner.append("            help=\"Enter 'AND', 'OR', or custom expression like '(1 AND 2) OR 3'\"\n")
new_scanner.append("        )\n")
new_scanner.append("    else:\n")
new_scanner.append("        st.session_state.scanner_logic = \"1\"\n")
new_scanner.append("else:\n")
new_scanner.append("    st.info(\"No conditions added yet. Use the dropdowns above to build conditions.\")\n")
new_scanner.append("\n")

# 6. Add footer: Scanner logic (from line 465 onwards in original)
new_scanner.append("# Use conditions for scanning\n")
new_scanner.append("conditions = st.session_state.scanner_conditions\n")
new_scanner.append("combination_logic = st.session_state.scanner_logic\n")
new_scanner.extend(scanner_lines[468:])  # Everything from scan button onwards

# Write the new file
with open('pages/Scanner_NEW.py', 'w', encoding='utf-8') as f:
    f.writelines(new_scanner)

print(f"Created new Scanner with {len(new_scanner)} lines")
print("File saved as: pages/Scanner_NEW.py")
print("Please review and then rename Scanner_NEW.py to Scanner.py")
