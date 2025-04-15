import React from 'react'
import { Table, TableBody, TableCell, TableHead, TableHeader, TableRow } from '@/components/ui/table'

interface PredictionsTableProps {
  predictions: { [key: string]: string }
}

export function PredictionsTable({ predictions }: PredictionsTableProps) {
  return (
    <div className="overflow-x-auto">
      <Table>
        <TableHeader>
          <TableRow>
            <TableHead>Class</TableHead>
            <TableHead>Probability</TableHead>
          </TableRow>
        </TableHeader>
        <TableBody>
          {Object.entries(predictions).map(([className, probability]) => (
            <TableRow key={className}>
              <TableCell>{className}</TableCell>
              <TableCell>{probability}</TableCell>
            </TableRow>
          ))}
        </TableBody>
      </Table>
    </div>
  )
}