VERS_1.0 {
  # Export JNI symbols.
  global:
    *CreateInstance*; 
    *predict*;
  # Hide everything else.
  local:
    *;
};
