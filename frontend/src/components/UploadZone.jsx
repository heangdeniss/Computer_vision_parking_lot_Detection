export default function UploadZone({
  tab, fileCount, videoName, inputRef, dragging,
  onFiles, onDragEnter, onDragOver, onDragLeave, onDrop,
}) {
  const isImages = tab === 'images'
  return (
    <div
      className={`dropzone ${dragging ? 'dragging' : ''}`}
      onDragEnter={onDragEnter}
      onDragOver={onDragOver}
      onDragLeave={onDragLeave}
      onDrop={onDrop}
    >
      <input
        ref={inputRef}
        id="file-input"
        type="file"
        multiple={isImages}
        accept={isImages ? 'image/*,.heic,.heif' : 'video/*'}
        onChange={(e) => { onFiles(e.target.files); e.target.value = '' }}
      />
      <label htmlFor="file-input">
        <strong>{isImages ? 'Drop images here' : 'Drop a video here'}</strong>
        <span>{isImages
          ? (fileCount ? `${fileCount} selected — drop more or click to add` : 'or click to browse (JPG, PNG, HEIC, multiple)')
          : (videoName || 'or click to browse (MP4, MOV, AVI)')}</span>
      </label>
    </div>
  )
}
