import React from 'react'

export default function InputField({ name, label, value, onChange, step = 'any' }){
  return (
    <div className="flex flex-col">
      <label className="text-sm font-medium text-gray-700 mb-1">{label || name}</label>
      <input
        name={name}
        type="number"
        step={step}
        value={value}
        onChange={onChange}
        className="border p-2 rounded w-full"
      />
    </div>
  )
}
